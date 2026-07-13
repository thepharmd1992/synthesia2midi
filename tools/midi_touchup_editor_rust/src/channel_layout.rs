use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NoteSpan {
    pub(crate) note_id: u64,
    pub(crate) pitch: u8,
    pub(crate) channel: u8,
    pub(crate) start_tick: u64,
    pub(crate) end_tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaneAssignment {
    pub(crate) index: usize,
    pub(crate) count: usize,
}

pub(crate) fn compute_lane_assignments(notes: &[NoteSpan]) -> HashMap<u64, LaneAssignment> {
    let mut by_pitch: BTreeMap<u8, Vec<NoteSpan>> = BTreeMap::new();
    for note in notes {
        by_pitch.entry(note.pitch).or_default().push(*note);
    }

    let mut result = HashMap::new();
    for pitch_notes in by_pitch.values_mut() {
        pitch_notes
            .sort_by_key(|note| (note.start_tick, note.end_tick, note.channel, note.note_id));

        let mut component_start = 0;
        while component_start < pitch_notes.len() {
            let mut component_end = component_start + 1;
            let mut max_end_tick = pitch_notes[component_start].end_tick;
            while component_end < pitch_notes.len()
                && pitch_notes[component_end].start_tick < max_end_tick
            {
                max_end_tick = max_end_tick.max(pitch_notes[component_end].end_tick);
                component_end += 1;
            }

            let channels: Vec<u8> = pitch_notes[component_start..component_end]
                .iter()
                .map(|note| note.channel)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            for note in &pitch_notes[component_start..component_end] {
                let index = channels.binary_search(&note.channel).unwrap_or(0);
                result.insert(
                    note.note_id,
                    LaneAssignment {
                        index,
                        count: channels.len().max(1),
                    },
                );
            }
            component_start = component_end;
        }
    }
    result
}

pub(crate) fn active_channels_at_tick(notes: &[NoteSpan], tick: u64) -> BTreeMap<u8, Vec<u8>> {
    let mut active: BTreeMap<u8, BTreeSet<u8>> = BTreeMap::new();
    for note in notes
        .iter()
        .filter(|note| note.start_tick <= tick && tick < note.end_tick)
    {
        active.entry(note.pitch).or_default().insert(note.channel);
    }
    active
        .into_iter()
        .map(|(pitch, channels)| (pitch, channels.into_iter().collect()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(id: u64, pitch: u8, channel: u8, start: u64, end: u64) -> NoteSpan {
        NoteSpan {
            note_id: id,
            pitch,
            channel,
            start_tick: start,
            end_tick: end,
        }
    }

    #[test]
    fn overlapping_channels_receive_stable_channel_ordered_lanes() {
        let notes = vec![
            span(1, 60, 3, 0, 100),
            span(2, 60, 0, 20, 80),
            span(3, 60, 2, 40, 120),
        ];
        let lanes = compute_lane_assignments(&notes);

        assert_eq!(lanes[&2], LaneAssignment { index: 0, count: 3 });
        assert_eq!(lanes[&3], LaneAssignment { index: 1, count: 3 });
        assert_eq!(lanes[&1], LaneAssignment { index: 2, count: 3 });
    }

    #[test]
    fn same_channel_duplicates_share_one_visual_lane() {
        let notes = vec![span(1, 60, 1, 0, 100), span(2, 60, 1, 20, 80)];
        let lanes = compute_lane_assignments(&notes);

        assert_eq!(lanes[&1], LaneAssignment { index: 0, count: 1 });
        assert_eq!(lanes[&2], LaneAssignment { index: 0, count: 1 });
    }

    #[test]
    fn non_overlapping_notes_keep_full_width() {
        let notes = vec![span(1, 60, 0, 0, 50), span(2, 60, 3, 50, 100)];
        let lanes = compute_lane_assignments(&notes);

        assert_eq!(lanes[&1].count, 1);
        assert_eq!(lanes[&2].count, 1);
    }

    #[test]
    fn active_channels_are_sorted_unique_per_pitch() {
        let notes = vec![
            span(1, 60, 3, 0, 100),
            span(2, 60, 0, 0, 100),
            span(3, 60, 3, 10, 90),
            span(4, 64, 2, 60, 120),
        ];

        let active = active_channels_at_tick(&notes, 50);

        assert_eq!(active[&60], vec![0, 3]);
        assert!(!active.contains_key(&64));
    }
}
