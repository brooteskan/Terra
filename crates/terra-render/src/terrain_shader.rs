//! Compile-time composition for the bounded and Infinite terrain shader variants.

/// Terrain sampling implementation baked into a terrain pipeline bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TerrainShaderVariant {
    Bounded,
    Infinite,
}

impl TerrainShaderVariant {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bounded => "bounded",
            Self::Infinite => "infinite",
        }
    }

    pub const fn shader_label(self) -> &'static str {
        match self {
            Self::Bounded => "terrain-bounded-shader",
            Self::Infinite => "terrain-infinite-shader",
        }
    }

    pub const fn terrain_pipeline_label(self) -> &'static str {
        match self {
            Self::Bounded => "terrain-bounded-pipe",
            Self::Infinite => "terrain-infinite-pipe",
        }
    }

    pub const fn ocean_pipeline_label(self) -> &'static str {
        match self {
            Self::Bounded => "ocean-bounded-pipe",
            Self::Infinite => "ocean-infinite-pipe",
        }
    }

    pub const fn wireframe_pipeline_label(self) -> &'static str {
        match self {
            Self::Bounded => "wireframe-bounded-pipe",
            Self::Infinite => "wireframe-infinite-pipe",
        }
    }
}

/// Compose one explicit WGSL source. Directives are comments, so the template
/// remains readable as WGSL while each output physically contains only the
/// selected sampling/resolution implementation.
pub(crate) fn compose(variant: TerrainShaderVariant) -> String {
    const TEMPLATE: &str = include_str!("shaders/terrain.wgsl");
    let mut selected: Option<TerrainShaderVariant> = None;
    let mut output = String::with_capacity(TEMPLATE.len());

    for line in TEMPLATE.lines() {
        match line.trim() {
            "// terra-variant:bounded" => {
                selected = Some(TerrainShaderVariant::Bounded);
            }
            "// terra-variant:infinite" => {
                selected = Some(TerrainShaderVariant::Infinite);
            }
            "// terra-variant:end" => selected = None,
            _ if selected.is_none() || selected == Some(variant) => {
                output.push_str(line);
                output.push('\n');
            }
            _ => {}
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_validate(variant: TerrainShaderVariant) -> naga::Module {
        let source = compose(variant);
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
            panic!(
                "{} terrain WGSL parse failed: {error}\n{source}",
                variant.name()
            )
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| {
            panic!(
                "{} terrain WGSL validation failed: {error:?}",
                variant.name()
            )
        });
        module
    }

    #[test]
    fn every_composed_variant_parses_and_validates() {
        for variant in [
            TerrainShaderVariant::Bounded,
            TerrainShaderVariant::Infinite,
        ] {
            let module = parse_and_validate(variant);
            for entry in ["vs_main", "fs_main", "vs_ocean", "fs_ocean", "fs_wireframe"] {
                assert!(
                    module
                        .entry_points
                        .iter()
                        .any(|candidate| candidate.name == entry),
                    "{} source is missing {entry}",
                    variant.name()
                );
            }
        }
    }

    #[test]
    fn bounded_source_excludes_infinite_addressing_and_runtime_selection() {
        let source = compose(TerrainShaderVariant::Bounded);
        for forbidden in [
            "Signed64Words",
            "sparse_hash",
            "lookup_tile_page_sparse",
            "resolve_height_infinite",
            "u.stream6.x",
        ] {
            assert!(
                !source.contains(forbidden),
                "bounded source retained {forbidden}"
            );
        }
    }

    #[test]
    fn infinite_source_retains_signed_sparse_resolution() {
        let source = compose(TerrainShaderVariant::Infinite);
        for required in [
            "Signed64Words",
            "sparse_hash",
            "lookup_tile_page_sparse",
            "resolve_height_infinite",
        ] {
            assert!(source.contains(required), "Infinite source lost {required}");
        }
    }

    #[test]
    fn variant_labels_are_distinct() {
        let bounded = TerrainShaderVariant::Bounded;
        let infinite = TerrainShaderVariant::Infinite;
        assert_ne!(bounded.shader_label(), infinite.shader_label());
        assert_ne!(
            bounded.terrain_pipeline_label(),
            infinite.terrain_pipeline_label()
        );
        assert_ne!(
            bounded.ocean_pipeline_label(),
            infinite.ocean_pipeline_label()
        );
        assert_ne!(
            bounded.wireframe_pipeline_label(),
            infinite.wireframe_pipeline_label()
        );
    }
}
