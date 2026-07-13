use std::collections::BTreeMap;

use eframe::egui::Color32;
use serde::Deserialize;

pub(crate) const COLOR_MAP_META_PREFIX: &[u8] = b"Synthesia2MIDI:color-map:v1:";
pub(crate) const MAX_COLOR_MAP_META_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChannelColors {
    pub(crate) natural: Option<[u8; 3]>,
    pub(crate) sharp_flat: Option<[u8; 3]>,
}

pub(crate) type ChannelColorMap = BTreeMap<u8, ChannelColors>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ColorMapPayload {
    channels: BTreeMap<String, ChannelColors>,
}

pub(crate) fn parse_color_map_text(text: &[u8]) -> Option<ChannelColorMap> {
    if text.len() > MAX_COLOR_MAP_META_BYTES || !text.starts_with(COLOR_MAP_META_PREFIX) {
        return None;
    }
    let body = std::str::from_utf8(&text[COLOR_MAP_META_PREFIX.len()..]).ok()?;
    let payload: ColorMapPayload = serde_json::from_str(body).ok()?;
    let mut result = ChannelColorMap::new();
    for (channel_text, colors) in payload.channels {
        let channel = channel_text.parse::<u8>().ok()?;
        if channel > 15 {
            return None;
        }
        if colors.natural.is_some() || colors.sharp_flat.is_some() {
            result.insert(channel, colors);
        }
    }
    (!result.is_empty()).then_some(result)
}

pub(crate) fn note_color(colors: &ChannelColorMap, channel: u8, pitch: u8) -> Color32 {
    let sharp_flat = is_black_key(pitch);
    let rgb = colors
        .get(&channel)
        .and_then(
            |entry| match (sharp_flat, entry.natural, entry.sharp_flat) {
                (false, Some(natural), _) => Some(natural),
                (true, _, Some(accidental)) => Some(accidental),
                (true, Some(natural), None) => Some(scale_rgb(natural, 0.72)),
                (false, None, Some(accidental)) => {
                    Some(blend_rgb(accidental, [255, 255, 255], 0.25))
                }
                _ => None,
            },
        )
        .unwrap_or_else(|| {
            let fallback = fallback_channel_rgb(channel);
            if sharp_flat {
                scale_rgb(fallback, 0.72)
            } else {
                fallback
            }
        });
    let readable = ensure_readable(rgb);
    Color32::from_rgba_unmultiplied(readable[0], readable[1], readable[2], 220)
}

fn is_black_key(pitch: u8) -> bool {
    matches!(pitch % 12, 1 | 3 | 6 | 8 | 10)
}

fn fallback_channel_rgb(channel: u8) -> [u8; 3] {
    const PALETTE: [[u8; 3]; 16] = [
        [90, 168, 255],
        [90, 220, 150],
        [255, 185, 96],
        [218, 112, 214],
        [86, 210, 225],
        [245, 112, 112],
        [222, 204, 92],
        [142, 125, 232],
        [80, 190, 170],
        [238, 142, 75],
        [188, 132, 230],
        [112, 188, 245],
        [148, 205, 95],
        [235, 105, 165],
        [196, 166, 92],
        [130, 175, 205],
    ];
    PALETTE[channel as usize % PALETTE.len()]
}

fn scale_rgb(rgb: [u8; 3], factor: f32) -> [u8; 3] {
    rgb.map(|component| ((component as f32 * factor).round()).clamp(0.0, 255.0) as u8)
}

fn blend_rgb(rgb: [u8; 3], target: [u8; 3], amount: f32) -> [u8; 3] {
    let amount = amount.clamp(0.0, 1.0);
    std::array::from_fn(|index| {
        ((rgb[index] as f32 * (1.0 - amount)) + (target[index] as f32 * amount))
            .round()
            .clamp(0.0, 255.0) as u8
    })
}

fn ensure_readable(rgb: [u8; 3]) -> [u8; 3] {
    let luminance = relative_luminance(rgb);
    if luminance < 0.18 {
        let amount = ((0.18 - luminance) / 0.18 * 0.45).clamp(0.08, 0.45);
        blend_rgb(rgb, [255, 255, 255], amount)
    } else if luminance > 0.88 {
        let amount = ((luminance - 0.88) / 0.12 * 0.35).clamp(0.08, 0.35);
        blend_rgb(rgb, [0, 0, 0], amount)
    } else {
        rgb
    }
}

fn relative_luminance(rgb: [u8; 3]) -> f32 {
    fn linear(component: u8) -> f32 {
        let value = component as f32 / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    0.2126 * linear(rgb[0]) + 0.7152 * linear(rgb[1]) + 0.0722 * linear(rgb[2])
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn parses_valid_version_one_color_map() {
        let parsed = parse_color_map_text(
            br#"Synthesia2MIDI:color-map:v1:{"channels":{"0":{"natural":[10,20,30],"sharp_flat":[4,8,12]},"3":{"natural":[200,100,40]}}}"#,
        )
        .expect("valid color map");

        assert_eq!(parsed.get(&0).unwrap().natural, Some([10, 20, 30]));
        assert_eq!(parsed.get(&0).unwrap().sharp_flat, Some([4, 8, 12]));
        assert_eq!(parsed.get(&3).unwrap().natural, Some([200, 100, 40]));
    }

    #[test]
    fn ignores_unknown_malformed_out_of_range_and_oversized_metadata() {
        assert!(parse_color_map_text(b"Synthesia2MIDI:color-map:v2:{}").is_none());
        assert!(parse_color_map_text(b"Synthesia2MIDI:color-map:v1:{bad").is_none());
        assert!(parse_color_map_text(
            br#"Synthesia2MIDI:color-map:v1:{"channels":{"16":{"natural":[1,2,3]}}}"#,
        )
        .is_none());
        assert!(parse_color_map_text(&vec![b'x'; MAX_COLOR_MAP_META_BYTES + 1]).is_none());
    }

    #[test]
    fn fallback_palette_distinguishes_first_four_channels() {
        let colors: HashSet<[u8; 4]> = (0..4)
            .map(|channel| note_color(&ChannelColorMap::new(), channel, 60).to_array())
            .collect();

        assert_eq!(colors.len(), 4);
    }

    #[test]
    fn explicit_morphologies_and_derived_pair_are_related_but_distinct() {
        let mut map = ChannelColorMap::new();
        map.insert(
            0,
            ChannelColors {
                natural: Some([90, 180, 220]),
                sharp_flat: Some([40, 90, 120]),
            },
        );
        assert_ne!(note_color(&map, 0, 60), note_color(&map, 0, 61));

        map.get_mut(&0).unwrap().sharp_flat = None;
        assert_ne!(note_color(&map, 0, 60), note_color(&map, 0, 61));
    }

    #[test]
    fn readable_midrange_explicit_color_is_preserved() {
        let mut map = ChannelColorMap::new();
        map.insert(
            2,
            ChannelColors {
                natural: Some([90, 180, 220]),
                sharp_flat: None,
            },
        );

        assert_eq!(
            note_color(&map, 2, 60),
            Color32::from_rgba_unmultiplied(90, 180, 220, 220)
        );
    }
}
