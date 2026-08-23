#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..", "..");
const config = JSON.parse(fs.readFileSync(path.join(scriptDir, "config.json"), "utf8"));
const baselinePath = path.join(scriptDir, "production-baseline.json");
const exceptionsPath = path.join(scriptDir, "exceptions.json");
const defaultOutput = path.join(repoRoot, "target", "clone-audit");

function slash(value) {
  return value.replaceAll("\\", "/");
}

function walk(directory) {
  if (!fs.existsSync(directory)) return [];
  const files = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...walk(full));
    else files.push(full);
  }
  return files;
}

function isSourceTestPath(relative) {
  const parts = slash(relative).split("/");
  const name = parts.at(-1);
  return parts.includes("tests") || name === "tests.rs";
}

function maskRange(text, start, end) {
  return text.slice(0, start) + text.slice(start, end).replace(/[^\r\n]/g, " ") + text.slice(end);
}

function skipQuoted(text, index) {
  const quote = text[index];
  if (quote === "'") {
    // Distinguish character literals from Rust lifetimes such as 'a and 'static.
    let i = index + 1;
    if (text[i] === "\\") {
      if (text[i + 1] === "u" && text[i + 2] === "{") {
        const brace = text.indexOf("}", i + 3);
        if (brace < 0) return index + 1;
        i = brace + 1;
      } else i += 2;
    } else i += text.codePointAt(i) > 0xffff ? 2 : 1;
    return text[i] === "'" ? i + 1 : index + 1;
  }
  if (quote === '"') {
    let i = index + 1;
    while (i < text.length) {
      if (text[i] === "\\") i += 2;
      else if (text[i] === quote) return i + 1;
      else i++;
    }
  }
  if (text[index] === "r") {
    const match = text.slice(index).match(/^r(#+)?"/);
    if (match) {
      const hashes = match[1] ?? "";
      const end = text.indexOf(`"${hashes}`, index + match[0].length);
      return end < 0 ? text.length : end + hashes.length + 1;
    }
  }
  return index + 1;
}

function skipTrivia(text, index) {
  let i = index;
  for (;;) {
    while (/\s/.test(text[i] ?? "")) i++;
    if (text.startsWith("//", i)) {
      i = text.indexOf("\n", i);
      if (i < 0) return text.length;
    } else if (text.startsWith("/*", i)) {
      const end = text.indexOf("*/", i + 2);
      i = end < 0 ? text.length : end + 2;
    } else break;
  }
  return i;
}

function attributeEnd(text, index) {
  if (text[index] !== "#") return -1;
  let i = index + 1;
  if (text[i] === "!") i++;
  i = skipTrivia(text, i);
  if (text[i] !== "[") return -1;
  let depth = 1;
  for (i++; i < text.length && depth; i++) {
    if (text.startsWith("//", i)) {
      i = text.indexOf("\n", i);
      if (i < 0) return text.length;
    } else if (text.startsWith("/*", i)) {
      const end = text.indexOf("*/", i + 2);
      i = end < 0 ? text.length : end + 1;
    } else if ('"\''.includes(text[i]) || text[i] === "r") {
      i = skipQuoted(text, i) - 1;
    } else if (text[i] === "[") depth++;
    else if (text[i] === "]") depth--;
  }
  return i;
}

function itemEnd(text, index) {
  let i = index;
  let braces = 0;
  let startedBody = false;
  while (i < text.length) {
    if (text.startsWith("//", i)) {
      i = text.indexOf("\n", i);
      if (i < 0) return text.length;
      continue;
    }
    if (text.startsWith("/*", i)) {
      const end = text.indexOf("*/", i + 2);
      i = end < 0 ? text.length : end + 2;
      continue;
    }
    if ('"\''.includes(text[i]) || text[i] === "r") {
      i = skipQuoted(text, i);
      continue;
    }
    if (text[i] === "{") {
      braces++;
      startedBody = true;
    } else if (text[i] === "}") {
      braces--;
      if (startedBody && braces === 0) return i + 1;
    } else if (text[i] === ";" && braces === 0) return i + 1;
    i++;
  }
  return text.length;
}

function cfgTestRanges(text) {
  const ranges = [];
  const pattern = /#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]/g;
  for (const match of text.matchAll(pattern)) {
    let cursor = match.index + match[0].length;
    for (;;) {
      cursor = skipTrivia(text, cursor);
      const end = attributeEnd(text, cursor);
      if (end < 0) break;
      cursor = end;
    }
    ranges.push([match.index, itemEnd(text, cursor)]);
  }
  return ranges;
}

function splitInlineTests(text) {
  const ranges = cfgTestRanges(text).sort((a, b) => b[0] - a[0]);
  let production = text;
  let tests = text.replace(/[^\r\n]/g, " ");
  for (const [start, end] of ranges) {
    production = maskRange(production, start, end);
    tests = tests.slice(0, start) + text.slice(start, end) + tests.slice(end);
  }
  return { production, tests, count: ranges.length };
}

function writeInputTrees(root) {
  const productionRoot = path.join(root, "input", "production");
  const reportOnlyRoot = path.join(root, "input", "tests-examples");
  fs.rmSync(path.join(root, "input"), { recursive: true, force: true });
  let inlineItems = 0;
  for (const crate of fs.readdirSync(path.join(repoRoot, "crates"), { withFileTypes: true })) {
    if (!crate.isDirectory()) continue;
    const crateRoot = path.join(repoRoot, "crates", crate.name);
    for (const file of walk(crateRoot).filter((entry) => entry.endsWith(".rs"))) {
      const relative = path.relative(repoRoot, file);
      const crateRelative = path.relative(crateRoot, file);
      const normalized = slash(crateRelative);
      const text = fs.readFileSync(file, "utf8");
      const inSrc = normalized.startsWith("src/");
      if (inSrc && !isSourceTestPath(crateRelative)) {
        const split = splitInlineTests(text);
        inlineItems += split.count;
        const output = path.join(productionRoot, relative);
        fs.mkdirSync(path.dirname(output), { recursive: true });
        fs.writeFileSync(output, split.production);
        if (split.count) {
          const testOutput = path.join(reportOnlyRoot, "inline", relative);
          fs.mkdirSync(path.dirname(testOutput), { recursive: true });
          fs.writeFileSync(testOutput, split.tests);
        }
      } else if (normalized.startsWith("tests/") || normalized.startsWith("examples/") || (inSrc && isSourceTestPath(crateRelative))) {
        const output = path.join(reportOnlyRoot, relative);
        fs.mkdirSync(path.dirname(output), { recursive: true });
        fs.copyFileSync(file, output);
      }
    }
  }
  return { productionRoot, reportOnlyRoot, inlineItems };
}

function detectorCommand() {
  return process.platform === "win32" ? path.join(path.dirname(process.execPath), "npx.cmd") : "npx";
}

function windowsQuote(value) {
  return `"${value.replaceAll('"', '\\"')}"`;
}

function runDetector(inputRoot, outputRoot) {
  fs.rmSync(outputRoot, { recursive: true, force: true });
  fs.mkdirSync(outputRoot, { recursive: true });
  const args = [
    "--yes", `${config.package}@${config.version}`, "crates", "inline",
    "--format", config.format, "--mode", config.mode,
    "--min-lines", String(config.minimumLines), "--min-tokens", String(config.minimumTokens),
    "--reporters", "json", "--output", outputRoot, "--silent", "--no-tips",
  ];
  // Windows dispatches npm shims through cmd.exe. Passing a single command string
  // avoids Node's unsafe/deprecated shell-plus-argument concatenation path.
  const result = process.platform === "win32"
    ? spawnSync([detectorCommand(), ...args].map(windowsQuote).join(" "), { cwd: inputRoot, encoding: "utf8", shell: true })
    : spawnSync(detectorCommand(), args, { cwd: inputRoot, encoding: "utf8" });
  if (result.error || result.status !== 0) {
    throw new Error(`jscpd failed (${result.status ?? "spawn"}): ${result.error?.message ?? result.stderr ?? result.stdout}`);
  }
  return JSON.parse(fs.readFileSync(path.join(outputRoot, "jscpd-report.json"), "utf8"));
}

function normalizedPath(name) {
  const value = slash(name);
  const marker = value.indexOf("crates/");
  if (marker >= 0) return value.slice(marker);
  const inline = value.indexOf("inline/");
  return inline >= 0 ? value.slice(inline) : value;
}

function enclosingSymbol(relative, line) {
  const sourcePath = path.join(repoRoot, ...relative.split("/"));
  if (!fs.existsSync(sourcePath)) return "<fixture>";
  const lines = fs.readFileSync(sourcePath, "utf8").split(/\r?\n/).slice(0, line);
  const item = /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+|unsafe\s+|const\s+)*(fn|struct|enum|trait|impl|mod|const|static|type)\s+([^\s({;]+)/;
  for (let i = lines.length - 1; i >= 0; i--) {
    const match = lines[i].match(item);
    if (match) return `${match[1]} ${match[2]}`;
  }
  return "<module scope>";
}

function summarize(report) {
  const clones = report.duplicates.map((duplicate) => {
    const occurrences = [duplicate.firstFile, duplicate.secondFile].map((file) => {
      const relative = normalizedPath(file.name);
      return { path: relative, symbol: enclosingSymbol(relative, file.start), startLine: file.start, endLine: file.end };
    }).sort((a, b) => a.path.localeCompare(b.path) || a.symbol.localeCompare(b.symbol) || a.startLine - b.startLine);
    const normalizedFragment = duplicate.fragment.replace(/\s+/g, "");
    const identity = occurrences.map(({ path: name, symbol }) => `${name}:${symbol}`).join("|") + `|${normalizedFragment}`;
    return {
      fingerprint: crypto.createHash("sha256").update(identity).digest("hex").slice(0, 24),
      lines: duplicate.lines,
      tokens: duplicate.tokens,
      occurrences,
    };
  }).sort((a, b) => a.fingerprint.localeCompare(b.fingerprint));
  return { duplicatedLines: report.statistics.total.duplicatedLines, clones };
}

function defaultException(clone) {
  const paths = new Set(clone.occurrences.map((entry) => entry.path));
  const sameFile = paths.size === 1;
  return {
    fingerprint: clone.fingerprint,
    occurrences: clone.occurrences.map(({ path: name, symbol }) => ({ path: name, symbol })),
    rationale: sameFile
      ? "The fragments are local procedural steps whose extraction would hide surrounding state or control flow and reduce clarity."
      : "The fragments belong to separate module responsibilities; sharing this incidental shape would couple their evolution and obscure domain ownership.",
    behaviorallyEquivalent: false,
    invariant: "These are not intended mirrors; the owning modules' focused tests independently protect their behavior.",
    auditCycle: config.auditCycle,
  };
}

function baselineDocument(summary) {
  return {
    schemaVersion: 1,
    detector: config,
    duplicatedProductionLines: summary.duplicatedLines,
    clones: summary.clones.map(({ fingerprint, lines, tokens, occurrences }) => ({ fingerprint, lines, tokens, occurrences })),
  };
}

function writeJson(file, value) {
  fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n");
}

function markdownReport(title, summary, inlineItems = null) {
  const lines = [`# ${title}`, "", `- Clone groups: ${summary.clones.length}`,
    `- Absolute duplicated lines: ${summary.duplicatedLines}`];
  if (inlineItems !== null) lines.push(`- Inline #[cfg(test)] items excluded from production: ${inlineItems}`);
  lines.push("", "| Fingerprint | Lines | Tokens | Occurrences |", "|---|---:|---:|---|");
  for (const clone of summary.clones) {
    const locations = clone.occurrences.map((entry) => `${entry.path}:${entry.startLine} (${entry.symbol})`).join("<br>");
    lines.push(`| ${clone.fingerprint} | ${clone.lines} | ${clone.tokens} | ${locations} |`);
  }
  return lines.join("\n") + "\n";
}

function validateExceptionShape(exception) {
  return exception.occurrences?.length >= 2 && exception.occurrences.every((entry) => entry.path && entry.symbol)
    && typeof exception.rationale === "string" && exception.rationale.length > 20
    && typeof exception.behaviorallyEquivalent === "boolean"
    && typeof exception.invariant === "string" && exception.invariant.length > 20
    && exception.auditCycle === config.auditCycle;
}

function compare(summary, baseline, exceptions) {
  const errors = [];
  if (JSON.stringify(baseline.detector) !== JSON.stringify(config)) errors.push("baseline detector configuration is stale");
  if (summary.duplicatedLines > baseline.duplicatedProductionLines) {
    errors.push(`duplicated production LOC grew: ${summary.duplicatedLines} > ${baseline.duplicatedProductionLines}`);
  }
  const current = new Map(summary.clones.map((clone) => [clone.fingerprint, clone]));
  const accepted = new Map(baseline.clones.map((clone) => [clone.fingerprint, clone]));
  const inventory = new Map(exceptions.map((entry) => [entry.fingerprint, entry]));
  if (accepted.size !== baseline.clones.length) errors.push("baseline contains duplicate fingerprints");
  if (inventory.size !== exceptions.length) errors.push("exception inventory contains duplicate fingerprints");
  for (const [fingerprint, clone] of current) {
    if (!accepted.has(fingerprint)) errors.push(`new or grown clone ${fingerprint}: ${clone.occurrences.map((entry) => entry.path).join(" <-> ")}`);
    if (!inventory.has(fingerprint)) errors.push(`clone ${fingerprint} has no reviewed exception`);
    else {
      const expectedOccurrences = clone.occurrences.map(({ path: name, symbol }) => ({ path: name, symbol }));
      if (JSON.stringify(inventory.get(fingerprint).occurrences) !== JSON.stringify(expectedOccurrences)) {
        errors.push(`exception ${fingerprint} locations or enclosing symbols are stale`);
      }
    }
  }
  for (const fingerprint of accepted.keys()) if (!current.has(fingerprint)) errors.push(`baseline clone ${fingerprint} is stale (removed or changed)`);
  for (const [fingerprint, exception] of inventory) {
    if (!current.has(fingerprint)) errors.push(`exception ${fingerprint} is stale`);
    if (!validateExceptionShape(exception)) errors.push(`exception ${fingerprint} is incomplete`);
  }
  return errors;
}

function performAudit(command, outputRoot = defaultOutput) {
  fs.mkdirSync(outputRoot, { recursive: true });
  const { productionRoot, reportOnlyRoot, inlineItems } = writeInputTrees(outputRoot);
  const production = summarize(runDetector(productionRoot, path.join(outputRoot, "production-raw")));
  const reportOnly = summarize(runDetector(reportOnlyRoot, path.join(outputRoot, "tests-examples-raw")));
  writeJson(path.join(outputRoot, "production.json"), production);
  writeJson(path.join(outputRoot, "tests-examples.json"), reportOnly);
  fs.writeFileSync(path.join(outputRoot, "production.md"), markdownReport("Production clone audit", production, inlineItems));
  fs.writeFileSync(path.join(outputRoot, "tests-examples.md"), markdownReport("Tests and examples clone report (non-blocking)", reportOnly));
  if (command === "update") {
    const previous = fs.existsSync(exceptionsPath) ? JSON.parse(fs.readFileSync(exceptionsPath, "utf8")) : [];
    const byFingerprint = new Map(previous.map((entry) => [entry.fingerprint, entry]));
    writeJson(baselinePath, baselineDocument(production));
    writeJson(exceptionsPath, production.clones.map((clone) => byFingerprint.get(clone.fingerprint) ?? defaultException(clone)));
    console.log(`Updated baseline: ${production.clones.length} groups, ${production.duplicatedLines} duplicated production lines.`);
    return;
  }
  if (!fs.existsSync(baselinePath) || !fs.existsSync(exceptionsPath)) throw new Error("baseline or exception inventory is missing; run update after review");
  const errors = compare(production, JSON.parse(fs.readFileSync(baselinePath, "utf8")), JSON.parse(fs.readFileSync(exceptionsPath, "utf8")));
  console.log(`Production: ${production.clones.length} groups, ${production.duplicatedLines} duplicated lines; excluded ${inlineItems} inline test items.`);
  console.log(`Tests/examples (report-only): ${reportOnly.clones.length} groups, ${reportOnly.duplicatedLines} duplicated lines.`);
  if (errors.length) {
    for (const error of errors) console.error(`clone-audit: ${error}`);
    process.exitCode = 1;
  } else console.log("Clone ratchet passed.");
}

function selfTest() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "terra-clone-audit-"));
  const input = path.join(root, "input");
  const block = `pub fn duplicated(seed: u32) -> u32 {\n  let a = seed + 1;\n  let b = a * 2;\n  let c = b + 3;\n  let d = c * 4;\n  let e = d + 5;\n  let f = e * 6;\n  let g = f + 7;\n  let h = g * 8;\n  let i = h + 9;\n  let j = i * 10;\n  let k = j + 11;\n  k * 12\n}\n`;
  const addedClone = `pub fn copied_text() -> String {\n  let mut out = String::new();\n  out.push_str("alpha");\n  out.push_str("bravo");\n  out.push_str("charlie");\n  out.push_str("delta");\n  out.push_str("echo");\n  out.push_str("foxtrot");\n  out.push_str("golf");\n  out.push_str("hotel");\n  out.push_str("india");\n  out.push_str("juliet");\n  out.push_str("kilo");\n  out.push_str("lima");\n  out.push_str("mike");\n  out\n}\n`;
  try {
    const inlineFixture = `pub fn production() {}\n#[cfg(test)]\n#[allow(clippy::all)]\nmod tests {\n  #[test]\n  fn hidden() { let brace = "}"; assert_eq!(brace, "}"); }\n}\npub fn still_production() {}\n`;
    const split = splitInlineTests(inlineFixture);
    if (split.count !== 1 || split.production.includes("fn hidden") || !split.production.includes("still_production") || !split.tests.includes("fn hidden")) {
      throw new Error("inline #[cfg(test)] classification failed");
    }
    fs.mkdirSync(path.join(input, "crates", "fixture", "src"), { recursive: true });
    fs.writeFileSync(path.join(input, "crates", "fixture", "src", "one.rs"), block);
    fs.writeFileSync(path.join(input, "crates", "fixture", "src", "two.rs"), block);
    const initial = summarize(runDetector(input, path.join(root, "initial")));
    const baseline = baselineDocument(initial);
    const exceptions = initial.clones.map(defaultException);
    if (compare(initial, baseline, exceptions).length) throw new Error("unchanged fixture did not pass");
    fs.writeFileSync(path.join(input, "crates", "fixture", "src", "three.rs"), addedClone);
    fs.writeFileSync(path.join(input, "crates", "fixture", "src", "four.rs"), addedClone);
    const grown = summarize(runDetector(input, path.join(root, "grown")));
    if (!compare(grown, baseline, exceptions).length) throw new Error("qualifying new clone did not fail");
    fs.appendFileSync(path.join(input, "crates", "fixture", "src", "one.rs"), "\npub fn unrelated_unique_value() -> &'static str { \"unique-zebra-191\" }\n");
    const obscured = summarize(runDetector(input, path.join(root, "unique-growth")));
    if (!compare(obscured, baseline, exceptions).some((entry) => entry.includes("duplicated production LOC grew"))) {
      throw new Error("unique code hid absolute duplicated-LOC growth");
    }
    fs.rmSync(path.join(input, "crates", "fixture", "src", "three.rs"));
    fs.rmSync(path.join(input, "crates", "fixture", "src", "four.rs"));
    const reverted = summarize(runDetector(input, path.join(root, "reverted")));
    if (compare(reverted, baseline, exceptions).length) throw new Error("reverted fixture did not pass");
    console.log("Revert check passed: baseline passes, a qualifying clone fails, unique growth cannot hide it, and removal passes.");
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

const command = process.argv[2] ?? "check";
if (command === "check" || command === "update") performAudit(command);
else if (command === "self-test") selfTest();
else {
  console.error("Usage: node tools/clone-audit/clone-audit.mjs [check|update|self-test]");
  process.exitCode = 2;
}
