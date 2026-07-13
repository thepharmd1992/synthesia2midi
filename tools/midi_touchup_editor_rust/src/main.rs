mod channel_layout;
mod color_map;

use std::collections::HashMap;
use std::fs;
#[cfg(any(target_os = "windows", target_os = "macos"))]
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::Parser;
#[cfg(any(target_os = "windows", target_os = "macos"))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::Sender;
#[cfg(any(target_os = "windows", target_os = "macos"))]
use crossbeam_channel::{unbounded, Receiver};
use eframe::egui::{self, Align2, Color32, FontId, Key, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use eframe::{App, Frame, NativeOptions};
use midly::{Fps, MetaMessage, MidiMessage, Smf, TrackEventKind};
use rfd::{FileDialog, MessageButtons, MessageDialog, MessageDialogResult, MessageLevel};
#[cfg(any(target_os = "windows", target_os = "macos"))]
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};
use serde::Serialize;

use channel_layout::{active_channels_at_tick, compute_lane_assignments, LaneAssignment, NoteSpan};
use color_map::{note_color, parse_color_map_text, ChannelColorMap};

const DEFAULT_TEMPO_US_PER_BEAT: u32 = 500_000;
const MIN_PITCH: u8 = 21;
const MAX_PITCH: u8 = 108;
const DEFAULT_VERTICAL_ZOOM: f32 = 1.0;
const MIN_VERTICAL_ZOOM: f32 = 0.35;
const MAX_VERTICAL_ZOOM: f32 = 6.0;
#[cfg(any(target_os = "windows", target_os = "macos"))]
const DEFAULT_SF2_FILENAME: &str = "TouchUpPiano.sf2";

#[derive(Clone, Copy, Debug)]
struct ToolbarMetrics {
    font_size: f32,
    button_width: f32,
    button_height: f32,
}

fn toolbar_metrics(available_width: f32) -> ToolbarMetrics {
    if available_width < 1100.0 {
        ToolbarMetrics {
            font_size: 18.0,
            button_width: 110.0,
            button_height: 38.0,
        }
    } else {
        ToolbarMetrics {
            font_size: 20.0,
            button_width: 150.0,
            button_height: 44.0,
        }
    }
}

fn pointer_targets_canvas(pointer_pos: Option<Pos2>, canvas_rect: Rect) -> bool {
    pointer_pos.is_some_and(|position| canvas_rect.contains(position))
}

#[derive(Parser, Debug)]
#[command(name = "midi-touchup-editor")]
#[command(about = "Standalone MIDI touch-up editor (falling bars only)")]
struct Cli {
    #[arg(long)]
    midi: PathBuf,

    #[arg(long, default_value_t = false)]
    result_json: bool,

    #[arg(long, default_value = "neothesia")]
    theme: String,

    #[arg(long)]
    sf2: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
struct EditorResult {
    status: String,
    source_path: String,
    saved_path: Option<String>,
    message: String,
}

impl EditorResult {
    fn saved(source_path: &Path, saved_path: &Path, message: impl Into<String>) -> Self {
        Self {
            status: "saved".to_string(),
            source_path: source_path.to_string_lossy().to_string(),
            saved_path: Some(saved_path.to_string_lossy().to_string()),
            message: message.into(),
        }
    }

    fn cancelled(source_path: &Path) -> Self {
        Self {
            status: "cancelled".to_string(),
            source_path: source_path.to_string_lossy().to_string(),
            saved_path: None,
            message: "Editor closed without saving.".to_string(),
        }
    }

    fn error(source_path: &Path, message: impl Into<String>) -> Self {
        Self {
            status: "error".to_string(),
            source_path: source_path.to_string_lossy().to_string(),
            saved_path: None,
            message: message.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GridOption {
    Quarter,
    Eighth,
    Sixteenth,
    ThirtySecond,
}

impl GridOption {
    fn all() -> [GridOption; 4] {
        [
            GridOption::Quarter,
            GridOption::Eighth,
            GridOption::Sixteenth,
            GridOption::ThirtySecond,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            GridOption::Quarter => "1/4",
            GridOption::Eighth => "1/8",
            GridOption::Sixteenth => "1/16",
            GridOption::ThirtySecond => "1/32",
        }
    }

    fn divisor(self) -> u16 {
        match self {
            GridOption::Quarter => 1,
            GridOption::Eighth => 2,
            GridOption::Sixteenth => 4,
            GridOption::ThirtySecond => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SnapTarget {
    Grid,
    Onset,
}

impl SnapTarget {
    fn all() -> [SnapTarget; 2] {
        [SnapTarget::Grid, SnapTarget::Onset]
    }

    fn label(self) -> &'static str {
        match self {
            SnapTarget::Grid => "Snap to Grid",
            SnapTarget::Onset => "Snap to Onset",
        }
    }
}

#[derive(Clone, Debug)]
struct EditableNote {
    note_id: u64,
    track_index: usize,
    channel: u8,
    pitch: u8,
    start_tick: u64,
    end_tick: u64,
    velocity_on: u8,
    velocity_off: u8,
    key_lane_unlocked: bool,
}

fn note_spans_for(notes: &[EditableNote]) -> Vec<NoteSpan> {
    notes
        .iter()
        .map(|note| NoteSpan {
            note_id: note.note_id,
            pitch: note.pitch,
            channel: note.channel,
            start_tick: note.start_tick,
            end_tick: note.end_tick,
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct TempoEvent {
    tick: u64,
    us_per_beat: u32,
}

#[derive(Clone, Copy, Debug)]
struct TempoSegment {
    start_tick: u64,
    start_sec: f64,
    us_per_beat: u32,
}

#[derive(Clone, Debug)]
struct TempoMap {
    ticks_per_beat: u16,
    events: Vec<TempoEvent>,
    segments: Vec<TempoSegment>,
}

impl TempoMap {
    fn from_events(ticks_per_beat: u16, default_tempo: u32, mut events: Vec<TempoEvent>) -> Self {
        let tpb = ticks_per_beat.max(1);
        events.sort_by_key(|e| e.tick);

        let mut deduped: Vec<TempoEvent> = Vec::new();
        for event in events.into_iter() {
            if let Some(last) = deduped.last_mut() {
                if last.tick == event.tick {
                    *last = event;
                    continue;
                }
            }
            deduped.push(event);
        }

        if deduped.is_empty() || deduped[0].tick > 0 {
            deduped.insert(
                0,
                TempoEvent {
                    tick: 0,
                    us_per_beat: default_tempo.max(1),
                },
            );
        } else if deduped[0].tick == 0 {
            deduped[0].us_per_beat = deduped[0].us_per_beat.max(1);
        }

        let mut segments: Vec<TempoSegment> = Vec::new();
        let mut running_sec = 0.0;
        for (idx, event) in deduped.iter().enumerate() {
            if idx > 0 {
                let prev = deduped[idx - 1];
                let prev_tps = tick_rate_per_sec(tpb, prev.us_per_beat);
                let delta_tick = event.tick.saturating_sub(prev.tick) as f64;
                running_sec += delta_tick / prev_tps;
            }
            segments.push(TempoSegment {
                start_tick: event.tick,
                start_sec: running_sec,
                us_per_beat: event.us_per_beat.max(1),
            });
        }

        Self {
            ticks_per_beat: tpb,
            events: deduped,
            segments,
        }
    }

    fn default(ticks_per_beat: u16) -> Self {
        Self::from_events(
            ticks_per_beat,
            DEFAULT_TEMPO_US_PER_BEAT,
            vec![TempoEvent {
                tick: 0,
                us_per_beat: DEFAULT_TEMPO_US_PER_BEAT,
            }],
        )
    }

    fn tick_to_sec(&self, tick: f64) -> f64 {
        if self.segments.is_empty() {
            return 0.0;
        }
        let clamped_tick = tick.max(0.0);
        let idx = self.find_segment_for_tick(clamped_tick);
        let seg = self.segments[idx];
        let tps = tick_rate_per_sec(self.ticks_per_beat, seg.us_per_beat);
        seg.start_sec + ((clamped_tick - seg.start_tick as f64) / tps)
    }

    fn sec_to_tick(&self, sec: f64) -> f64 {
        if self.segments.is_empty() {
            return 0.0;
        }
        let clamped_sec = sec.max(0.0);
        let idx = self.find_segment_for_sec(clamped_sec);
        let seg = self.segments[idx];
        let tps = tick_rate_per_sec(self.ticks_per_beat, seg.us_per_beat);
        seg.start_tick as f64 + ((clamped_sec - seg.start_sec) * tps)
    }

    fn find_segment_for_tick(&self, tick: f64) -> usize {
        match self
            .segments
            .binary_search_by(|seg| seg.start_tick.cmp(&(tick as u64)))
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        }
    }

    fn find_segment_for_sec(&self, sec: f64) -> usize {
        match self.segments.binary_search_by(|seg| {
            seg.start_sec
                .partial_cmp(&sec)
                .unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        }
    }
}

fn tick_rate_per_sec(ticks_per_beat: u16, us_per_beat: u32) -> f64 {
    let tpb = ticks_per_beat.max(1) as f64;
    let tempo = us_per_beat.max(1) as f64;
    tpb * (1_000_000.0 / tempo)
}

#[derive(Clone, Debug)]
struct PreservedEvent {
    tick: u64,
    order: u32,
    raw_bytes: Vec<u8>,
    is_end_of_track: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct MidiDocument {
    source_path: PathBuf,
    format_u16: u16,
    division_u16: u16,
    ticks_per_beat: u16,
    tempo_us_per_beat: u32,
    tempo_map: TempoMap,
    notes: Vec<EditableNote>,
    channel_colors: ChannelColorMap,
    preserved_tracks: Vec<Vec<PreservedEvent>>,
    max_tick: u64,
    next_note_id: u64,
    dirty: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DragMode {
    Move,
    ResizeEnd,
}

#[derive(Clone, Debug)]
struct DragState {
    note_id: u64,
    mode: DragMode,
    pointer_origin: Pos2,
    original_note: EditableNote,
}

#[derive(Clone, Debug)]
enum EditCommand {
    Delete {
        note: EditableNote,
        index: usize,
    },
    Update {
        note_id: u64,
        before: EditableNote,
        after: EditableNote,
    },
    Transpose {
        changes: Vec<PitchChange>,
        delta_octaves: i8,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PitchChange {
    note_id: u64,
    before: u8,
    after: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OctaveShiftBlock {
    BelowPiano { pitch: u8 },
    AbovePiano { pitch: u8 },
}

fn plan_octave_shift<I>(notes: I, delta_octaves: i8) -> Result<Vec<PitchChange>, OctaveShiftBlock>
where
    I: IntoIterator<Item = (u64, u8)>,
{
    assert!(matches!(delta_octaves, -1 | 1));
    let notes: Vec<(u64, u8)> = notes.into_iter().collect();
    let semitones = delta_octaves as i16 * 12;

    if delta_octaves < 0 {
        if let Some(lowest) = notes.iter().map(|(_, pitch)| *pitch).min() {
            if lowest as i16 + semitones < MIN_PITCH as i16 {
                return Err(OctaveShiftBlock::BelowPiano { pitch: lowest });
            }
        }
    } else if let Some(highest) = notes.iter().map(|(_, pitch)| *pitch).max() {
        if highest as i16 + semitones > MAX_PITCH as i16 {
            return Err(OctaveShiftBlock::AbovePiano { pitch: highest });
        }
    }

    Ok(notes
        .into_iter()
        .map(|(note_id, before)| PitchChange {
            note_id,
            before,
            after: (before as i16 + semitones) as u8,
        })
        .collect())
}

fn apply_pitch_changes(
    notes: &mut [EditableNote],
    changes: &[PitchChange],
    forward: bool,
) -> usize {
    let target_pitches: HashMap<u64, u8> = changes
        .iter()
        .map(|change| {
            (
                change.note_id,
                if forward { change.after } else { change.before },
            )
        })
        .collect();
    let mut updated = 0;
    for note in notes {
        if let Some(pitch) = target_pitches.get(&note.note_id) {
            note.pitch = *pitch;
            updated += 1;
        }
    }
    updated
}

fn octave_offset_label(offset: i8) -> String {
    if offset == 0 {
        "0".to_string()
    } else {
        format!("{offset:+}")
    }
}

fn midi_pitch_label(pitch: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = pitch as i16 / 12 - 1;
    format!("{}{}", NAMES[pitch as usize % 12], octave)
}

fn octave_shift_block_message(block: OctaveShiftBlock) -> String {
    match block {
        OctaveShiftBlock::BelowPiano { pitch } => format!(
            "Cannot shift down one octave because the lowest note is {} (MIDI {}). All notes must remain between A0 and C8.",
            midi_pitch_label(pitch),
            pitch
        ),
        OctaveShiftBlock::AbovePiano { pitch } => format!(
            "Cannot shift up one octave because the highest note is {} (MIDI {}). All notes must remain between A0 and C8.",
            midi_pitch_label(pitch),
            pitch
        ),
    }
}

#[derive(Clone, Debug)]
struct RawEvent {
    tick: u64,
    order: u64,
    raw_bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct FallingViewLayout {
    rect: Rect,
    strike_y: f32,
    keyboard_rect: Rect,
    pitch_min: u8,
    pitch_max: u8,
    horizon_ticks: f64,
    px_per_tick: f64,
}

#[derive(Clone, Debug)]
struct TimelineDragState {
    start_x: f32,
    start_tick: f64,
    fine_mode: bool,
}

#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
#[derive(Clone, Debug)]
enum SongEventKind {
    NoteOn {
        channel: u8,
        pitch: u8,
        velocity: u8,
    },
    NoteOff {
        channel: u8,
        pitch: u8,
    },
}

#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
#[derive(Clone, Debug)]
struct SongEvent {
    tick: u64,
    sec: f64,
    order: u8,
    kind: SongEventKind,
}

#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
#[derive(Clone, Debug)]
struct SongRenderData {
    tempo_map: TempoMap,
    notes: Vec<EditableNote>,
    events: Vec<SongEvent>,
    max_tick: u64,
    max_sec: f64,
}

#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
impl SongRenderData {
    fn from_document(doc: &MidiDocument) -> Self {
        let mut events: Vec<SongEvent> = Vec::new();
        for note in doc.notes.iter() {
            events.push(SongEvent {
                tick: note.end_tick,
                sec: doc.tempo_map.tick_to_sec(note.end_tick as f64),
                order: 0,
                kind: SongEventKind::NoteOff {
                    channel: note.channel,
                    pitch: note.pitch,
                },
            });
            events.push(SongEvent {
                tick: note.start_tick,
                sec: doc.tempo_map.tick_to_sec(note.start_tick as f64),
                order: 1,
                kind: SongEventKind::NoteOn {
                    channel: note.channel,
                    pitch: note.pitch,
                    velocity: note.velocity_on.max(1),
                },
            });
        }

        events.sort_by(|a, b| (a.tick, a.order).cmp(&(b.tick, b.order)));

        Self {
            tempo_map: doc.tempo_map.clone(),
            notes: doc.notes.clone(),
            events,
            max_tick: doc.max_tick,
            max_sec: doc.tempo_map.tick_to_sec(doc.max_tick as f64),
        }
    }

    fn event_index_after_tick(&self, tick: u64) -> usize {
        self.events.partition_point(|event| event.tick <= tick)
    }

    fn active_notes_at_tick(&self, tick: u64) -> Vec<(u8, u8, u8)> {
        self.notes
            .iter()
            .filter(|n| n.start_tick <= tick && tick < n.end_tick)
            .map(|n| (n.channel, n.pitch, n.velocity_on.max(1)))
            .collect()
    }
}

#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
#[derive(Clone, Debug)]
enum AudioCommand {
    Play,
    Pause,
    SeekTick(u64),
    SetSpeed(f32),
    SetMute(bool),
    SetVolume(f32),
    PreviewNote {
        pitch: u8,
        velocity: u8,
        duration_ms: u32,
    },
    LoadSong(SongRenderData),
    Shutdown,
}

#[derive(Debug, Default)]
struct AudioTelemetry {
    playhead_tick_bits: AtomicU64,
    meter_bits: AtomicU32,
}

#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
impl AudioTelemetry {
    fn set_playhead_tick(&self, tick: f64) {
        self.playhead_tick_bits
            .store(tick.max(0.0).to_bits(), Ordering::Relaxed);
    }

    fn playhead_tick(&self) -> f64 {
        f64::from_bits(self.playhead_tick_bits.load(Ordering::Relaxed))
    }

    fn set_meter(&self, meter: f32) {
        self.meter_bits
            .store(meter.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    fn meter(&self) -> f32 {
        f32::from_bits(self.meter_bits.load(Ordering::Relaxed))
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
#[derive(Debug)]
struct PreviewVoice {
    channel: u8,
    pitch: u8,
    remaining_samples: u32,
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
struct AudioRuntime {
    synth: Synthesizer,
    song: SongRenderData,
    event_index: usize,
    playback_sec: f64,
    playing: bool,
    speed: f32,
    muted: bool,
    volume: f32,
    preview_voices: Vec<PreviewVoice>,
    sample_rate: f64,
}

struct AudioEngineHandle {
    sender: Sender<AudioCommand>,
    telemetry: Arc<AudioTelemetry>,
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    stream: cpal::Stream,
}

struct MidiTouchupApp {
    document: MidiDocument,
    note_spans: Vec<NoteSpan>,
    note_lanes: HashMap<u64, LaneAssignment>,
    selected_note_id: Option<u64>,
    context_menu_note_id: Option<u64>,
    drag_state: Option<DragState>,
    drag_preview: Option<EditableNote>,
    undo_stack: Vec<EditCommand>,
    redo_stack: Vec<EditCommand>,
    octave_offset: i8,
    snap_enabled: bool,
    snap_target: SnapTarget,
    grid_option: GridOption,
    speed_label: String,
    speed_factor: f32,
    playing: bool,
    playhead_tick: f64,
    vertical_zoom: f32,
    timeline_drag_state: Option<TimelineDragState>,
    timeline_scrub_audio_mute_active: bool,
    last_frame_instant: Instant,
    show_close_prompt: bool,
    allow_close_without_prompt_once: bool,
    audio_engine: Option<AudioEngineHandle>,
    audio_muted: bool,
    audio_volume: f32,
    theme_applied: bool,
    result_state: Arc<Mutex<EditorResult>>,
    status_line: String,
}

impl MidiTouchupApp {
    fn new(
        midi_path: &Path,
        _theme_name: String,
        sf2_override: Option<PathBuf>,
        result_state: Arc<Mutex<EditorResult>>,
    ) -> Result<Self, String> {
        let document = load_midi_document(midi_path)?;
        let audio_engine = AudioEngineHandle::new(sf2_override, &document).ok();
        Ok(Self::from_document(document, audio_engine, result_state))
    }

    fn from_document(
        document: MidiDocument,
        audio_engine: Option<AudioEngineHandle>,
        result_state: Arc<Mutex<EditorResult>>,
    ) -> Self {
        let note_spans = note_spans_for(&document.notes);
        let note_lanes = compute_lane_assignments(&note_spans);

        Self {
            document,
            note_spans,
            note_lanes,
            selected_note_id: None,
            context_menu_note_id: None,
            drag_state: None,
            drag_preview: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            octave_offset: 0,
            snap_enabled: true,
            snap_target: SnapTarget::Grid,
            grid_option: GridOption::ThirtySecond,
            speed_label: "1.0x".to_string(),
            speed_factor: 1.0,
            playing: false,
            playhead_tick: 0.0,
            vertical_zoom: DEFAULT_VERTICAL_ZOOM,
            timeline_drag_state: None,
            timeline_scrub_audio_mute_active: false,
            last_frame_instant: Instant::now(),
            show_close_prompt: false,
            allow_close_without_prompt_once: false,
            audio_engine,
            audio_muted: false,
            audio_volume: 1.0,
            theme_applied: false,
            result_state,
            status_line: "Ready".to_string(),
        }
    }

    fn active_note_channels_at_playhead(&self) -> std::collections::BTreeMap<u8, Vec<u8>> {
        let t = self.playhead_tick.max(0.0) as u64;
        active_channels_at_tick(&self.note_spans, t)
    }

    fn recompute_channel_layout(&mut self) {
        self.note_spans = note_spans_for(&self.document.notes);
        self.note_lanes = compute_lane_assignments(&self.note_spans);
    }

    fn grid_ticks(&self) -> u64 {
        let tpb = self.document.ticks_per_beat.max(1) as u64;
        let divisor = self.grid_option.divisor().max(1) as u64;
        (tpb / divisor).max(1)
    }

    fn snap_delta_tick(&self, raw_delta: i64) -> i64 {
        if !self.snap_enabled {
            return raw_delta;
        }
        let grid = self.grid_ticks() as i64;
        if grid <= 1 {
            return raw_delta;
        }
        if raw_delta >= 0 {
            ((raw_delta + grid / 2) / grid) * grid
        } else {
            -(((-raw_delta + grid / 2) / grid) * grid)
        }
    }

    fn snap_to_nearest_onset_tick(&self, target_tick: u64, exclude_note_id: Option<u64>) -> u64 {
        let mut best_tick: Option<u64> = None;
        let mut best_dist = u64::MAX;

        for note in self.document.notes.iter() {
            if exclude_note_id == Some(note.note_id) {
                continue;
            }
            let dist = note.start_tick.abs_diff(target_tick);
            if dist < best_dist
                || (dist == best_dist
                    && best_tick.map_or(true, |current| note.start_tick < current))
            {
                best_dist = dist;
                best_tick = Some(note.start_tick);
            }
        }

        best_tick.unwrap_or(target_tick)
    }

    fn recompute_max_tick(&mut self) {
        let notes_max = self
            .document
            .notes
            .iter()
            .map(|n| n.end_tick)
            .max()
            .unwrap_or(self.document.ticks_per_beat as u64 * 8);
        self.document.max_tick = notes_max.max(self.document.ticks_per_beat as u64 * 8);
    }

    fn load_new_midi(&mut self, midi_path: &Path) {
        if self.document.dirty {
            let answer = MessageDialog::new()
                .set_level(MessageLevel::Info)
                .set_title("Unsaved Changes")
                .set_description("Discard current unsaved edits and open another MIDI file?")
                .set_buttons(MessageButtons::YesNo)
                .show();
            if answer != MessageDialogResult::Yes {
                return;
            }
        }

        match load_midi_document(midi_path) {
            Ok(doc) => {
                self.document = doc;
                self.recompute_channel_layout();
                self.selected_note_id = None;
                self.context_menu_note_id = None;
                self.drag_state = None;
                self.drag_preview = None;
                self.undo_stack.clear();
                self.redo_stack.clear();
                self.octave_offset = 0;
                self.set_playing(false);
                self.set_playhead_tick(0.0, true);
                self.timeline_drag_state = None;
                self.end_timeline_scrub_audio_mute();
                self.status_line = format!("Opened {}", midi_path.display());
                self.refresh_audio_song_data();
                if let Ok(mut result) = self.result_state.lock() {
                    *result = EditorResult::cancelled(&self.document.source_path);
                }
            }
            Err(err) => {
                self.status_line = format!("Open failed: {err}");
                let _ = MessageDialog::new()
                    .set_level(MessageLevel::Error)
                    .set_title("Open MIDI Failed")
                    .set_description(&err)
                    .set_buttons(MessageButtons::Ok)
                    .show();
            }
        }
    }

    fn save_touchup(&mut self) {
        match save_midi_document(&self.document) {
            Ok(saved_path) => {
                self.document.dirty = false;
                self.status_line = format!("Saved {}", saved_path.display());
                if let Ok(mut result) = self.result_state.lock() {
                    *result = EditorResult::saved(
                        &self.document.source_path,
                        &saved_path,
                        "Touch-up MIDI saved successfully.",
                    );
                }
                let _ = MessageDialog::new()
                    .set_level(MessageLevel::Info)
                    .set_title("Touch-Up Saved")
                    .set_description(format!("Saved to:\n{}", saved_path.display()))
                    .set_buttons(MessageButtons::Ok)
                    .show();
            }
            Err(err) => {
                self.status_line = format!("Save failed: {err}");
                if let Ok(mut result) = self.result_state.lock() {
                    *result = EditorResult::error(&self.document.source_path, err.clone());
                }
                let _ = MessageDialog::new()
                    .set_level(MessageLevel::Error)
                    .set_title("Save Failed")
                    .set_description(&err)
                    .set_buttons(MessageButtons::Ok)
                    .show();
            }
        }
    }

    fn note_by_id(&self, note_id: u64) -> Option<&EditableNote> {
        self.document.notes.iter().find(|n| n.note_id == note_id)
    }

    fn note_by_id_mut(&mut self, note_id: u64) -> Option<&mut EditableNote> {
        self.document
            .notes
            .iter_mut()
            .find(|n| n.note_id == note_id)
    }

    fn clamp_tick(&self, tick: f64) -> f64 {
        tick.clamp(0.0, self.document.max_tick as f64)
    }

    fn tick_from_timeline_x(&self, rect: Rect, x: f32) -> f64 {
        if rect.width() <= 1.0 {
            return 0.0;
        }
        let normalized = ((x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        self.document.max_tick as f64 * normalized as f64
    }

    fn playback_seconds(&self) -> f64 {
        self.document.tempo_map.tick_to_sec(self.playhead_tick)
    }

    fn set_playhead_tick(&mut self, tick: f64, notify_audio: bool) {
        self.playhead_tick = self.clamp_tick(tick);
        if notify_audio {
            if let Some(audio) = &self.audio_engine {
                audio.send(AudioCommand::SeekTick(self.playhead_tick.max(0.0) as u64));
            }
        }
    }

    fn set_playing(&mut self, playing: bool) {
        self.playing = playing;
        if let Some(audio) = &self.audio_engine {
            if playing {
                audio.send(AudioCommand::Play);
            } else {
                audio.send(AudioCommand::Pause);
            }
        }
    }

    fn set_speed(&mut self, speed: f32, speed_label: String) {
        self.speed_factor = speed.max(0.1);
        self.speed_label = speed_label;
        if let Some(audio) = &self.audio_engine {
            audio.send(AudioCommand::SetSpeed(self.speed_factor));
        }
    }

    fn set_audio_mute(&mut self, muted: bool) {
        self.audio_muted = muted;
        if let Some(audio) = &self.audio_engine {
            audio.send(AudioCommand::SetMute(
                self.audio_muted || self.timeline_scrub_audio_mute_active,
            ));
        }
    }

    fn set_audio_volume(&mut self, volume: f32) {
        self.audio_volume = volume.clamp(0.0, 1.0);
        if let Some(audio) = &self.audio_engine {
            audio.send(AudioCommand::SetVolume(self.audio_volume));
        }
    }

    fn refresh_audio_song_data(&mut self) {
        if let Some(audio) = &self.audio_engine {
            let song = SongRenderData::from_document(&self.document);
            audio.send(AudioCommand::LoadSong(song));
            audio.send(AudioCommand::SeekTick(self.playhead_tick.max(0.0) as u64));
            audio.send(AudioCommand::SetSpeed(self.speed_factor));
            audio.send(AudioCommand::SetMute(
                self.audio_muted || self.timeline_scrub_audio_mute_active,
            ));
            audio.send(AudioCommand::SetVolume(self.audio_volume));
            if self.playing {
                audio.send(AudioCommand::Play);
            } else {
                audio.send(AudioCommand::Pause);
            }
        }
    }

    fn begin_timeline_scrub_audio_mute(&mut self) {
        if self.timeline_scrub_audio_mute_active {
            return;
        }
        self.timeline_scrub_audio_mute_active = true;
        if let Some(audio) = &self.audio_engine {
            audio.send(AudioCommand::SetMute(true));
        }
    }

    fn end_timeline_scrub_audio_mute(&mut self) {
        if !self.timeline_scrub_audio_mute_active {
            return;
        }
        self.timeline_scrub_audio_mute_active = false;
        if let Some(audio) = &self.audio_engine {
            audio.send(AudioCommand::SetMute(self.audio_muted));
        }
    }

    fn audition_pitch(&self, pitch: u8, velocity: u8, duration_ms: u32) {
        if let Some(audio) = &self.audio_engine {
            audio.send(AudioCommand::PreviewNote {
                pitch,
                velocity: velocity.max(1),
                duration_ms: duration_ms.max(20),
            });
        }
    }

    fn push_command(&mut self, cmd: EditCommand) {
        self.apply_command(&cmd, true);
        self.undo_stack.push(cmd);
        self.redo_stack.clear();
        self.document.dirty = true;
        self.recompute_max_tick();
        self.recompute_channel_layout();
        self.refresh_audio_song_data();
    }

    fn apply_command(&mut self, cmd: &EditCommand, forward: bool) {
        match cmd {
            EditCommand::Delete { note, index } => {
                if forward {
                    self.document.notes.retain(|n| n.note_id != note.note_id);
                    if self.selected_note_id == Some(note.note_id) {
                        self.selected_note_id = None;
                    }
                } else {
                    let idx = (*index).min(self.document.notes.len());
                    self.document.notes.insert(idx, note.clone());
                }
            }
            EditCommand::Update {
                note_id,
                before,
                after,
            } => {
                let replacement = if forward { after } else { before };
                if let Some(target) = self.note_by_id_mut(*note_id) {
                    *target = replacement.clone();
                }
            }
            EditCommand::Transpose {
                changes,
                delta_octaves,
            } => {
                let updated = apply_pitch_changes(&mut self.document.notes, changes, forward);
                debug_assert_eq!(updated, changes.len());
                self.octave_offset += if forward {
                    *delta_octaves
                } else {
                    -*delta_octaves
                };
            }
        }
    }

    fn apply_octave_shift(&mut self, delta_octaves: i8) -> Result<(), OctaveShiftBlock> {
        let changes = plan_octave_shift(
            self.document
                .notes
                .iter()
                .map(|note| (note.note_id, note.pitch)),
            delta_octaves,
        )?;
        if changes.is_empty() {
            return Ok(());
        }
        self.set_playing(false);
        self.push_command(EditCommand::Transpose {
            changes,
            delta_octaves,
        });
        self.status_line = format!("Octave adjustment: {:+}", self.octave_offset);
        Ok(())
    }

    fn request_octave_shift(&mut self, delta_octaves: i8) {
        if let Err(block) = self.apply_octave_shift(delta_octaves) {
            let _ = MessageDialog::new()
                .set_level(MessageLevel::Info)
                .set_title("Octave Shift Blocked")
                .set_description(octave_shift_block_message(block))
                .set_buttons(MessageButtons::Ok)
                .show();
        }
    }

    fn undo(&mut self) {
        if let Some(cmd) = self.undo_stack.pop() {
            self.apply_command(&cmd, false);
            self.redo_stack.push(cmd);
            self.document.dirty = true;
            self.recompute_max_tick();
            self.recompute_channel_layout();
            self.refresh_audio_song_data();
            self.status_line = "Undo".to_string();
        }
    }

    fn redo(&mut self) {
        if let Some(cmd) = self.redo_stack.pop() {
            self.apply_command(&cmd, true);
            self.undo_stack.push(cmd);
            self.document.dirty = true;
            self.recompute_max_tick();
            self.recompute_channel_layout();
            self.refresh_audio_song_data();
            self.status_line = "Redo".to_string();
        }
    }

    fn delete_selected_note(&mut self) {
        let Some(note_id) = self.selected_note_id else {
            return;
        };
        let Some((index, note)) = self
            .document
            .notes
            .iter()
            .enumerate()
            .find(|(_, n)| n.note_id == note_id)
            .map(|(i, n)| (i, n.clone()))
        else {
            return;
        };

        self.push_command(EditCommand::Delete { note, index });
        if self.context_menu_note_id == Some(note_id) {
            self.context_menu_note_id = None;
        }
        self.status_line = "Deleted selected note".to_string();
    }

    fn start_drag(&mut self, pointer_pos: Pos2, layout: &FallingViewLayout, force_resize: bool) {
        let Some((note_id, mode)) = self.pick_note(pointer_pos, layout, force_resize) else {
            self.selected_note_id = None;
            self.drag_state = None;
            self.drag_preview = None;
            return;
        };
        self.selected_note_id = Some(note_id);
        if let Some(note) = self.note_by_id(note_id).cloned() {
            self.drag_state = Some(DragState {
                note_id,
                mode,
                pointer_origin: pointer_pos,
                original_note: note.clone(),
            });
            self.drag_preview = Some(note);
        }
    }

    fn update_drag(&mut self, pointer_pos: Pos2, layout: &FallingViewLayout) {
        let Some(state) = self.drag_state.clone() else {
            return;
        };

        let delta = pointer_pos - state.pointer_origin;
        let delta_tick = if layout.px_per_tick > 0.0 {
            (delta.y as f64 / layout.px_per_tick).round() as i64
        } else {
            0
        };

        let mut preview = state.original_note.clone();
        match state.mode {
            DragMode::Move => {
                let duration = (state.original_note.end_tick as i64
                    - state.original_note.start_tick as i64)
                    .max(1);
                // In falling-note coordinates, increasing y (dragging down) should move the note down.
                let unsnapped_start =
                    (state.original_note.start_tick as i64 - delta_tick).max(0) as u64;
                let snapped_start = if self.snap_enabled {
                    match self.snap_target {
                        SnapTarget::Grid => {
                            let snapped_shift = self.snap_delta_tick(-delta_tick);
                            let new_start =
                                (state.original_note.start_tick as i64 + snapped_shift).max(0);
                            new_start as u64
                        }
                        SnapTarget::Onset => {
                            self.snap_to_nearest_onset_tick(unsnapped_start, Some(state.note_id))
                        }
                    }
                } else {
                    unsnapped_start
                };
                let new_end = snapped_start.saturating_add(duration as u64);

                preview.start_tick = snapped_start;
                preview.end_tick = new_end.max(snapped_start + 1);
                if state.original_note.key_lane_unlocked {
                    preview.pitch = pitch_from_x(layout, pointer_pos.x);
                }
            }
            DragMode::ResizeEnd => {
                let snapped_shift = self.snap_delta_tick(-delta_tick);
                let mut new_end = state.original_note.end_tick as i64 + snapped_shift;
                new_end = new_end.max(state.original_note.start_tick as i64 + 1);
                preview.end_tick = (new_end as u64).max(preview.start_tick + 1);
            }
        }
        self.drag_preview = Some(preview);
    }

    fn finish_drag(&mut self) {
        let Some(state) = self.drag_state.take() else {
            self.drag_preview = None;
            return;
        };
        let Some(preview) = self.drag_preview.take() else {
            return;
        };
        let before = state.original_note;
        if before.start_tick == preview.start_tick
            && before.end_tick == preview.end_tick
            && before.pitch == preview.pitch
        {
            return;
        }
        self.push_command(EditCommand::Update {
            note_id: state.note_id,
            before,
            after: preview,
        });
        if let Some(updated) = self.note_by_id(state.note_id) {
            self.audition_pitch(updated.pitch, updated.velocity_on, 180);
        }
        self.status_line = "Note updated".to_string();
    }

    fn pick_note(
        &self,
        pointer_pos: Pos2,
        layout: &FallingViewLayout,
        force_resize: bool,
    ) -> Option<(u64, DragMode)> {
        let mut best: Option<(u64, DragMode, f32)> = None;
        for note in self.document.notes.iter() {
            let rect = self.note_rect(note, layout);
            // Tiny notes can become hard to grab; expand only the hit area (not the drawn rect).
            let min_pick_height = 14.0_f32;
            let pick_rect = if rect.height() < min_pick_height {
                let extra = (min_pick_height - rect.height()) * 0.5;
                Rect::from_min_max(
                    Pos2::new(rect.min.x, rect.min.y - extra),
                    Pos2::new(rect.max.x, rect.max.y + extra),
                )
            } else {
                rect
            };
            if !pick_rect.contains(pointer_pos) {
                continue;
            }

            // Mode is explicit: Shift+drag resizes, normal drag moves.
            let mode = if force_resize {
                DragMode::ResizeEnd
            } else {
                DragMode::Move
            };
            let score = rect.max.y;
            if let Some((_, _, best_score)) = best {
                if score > best_score {
                    best = Some((note.note_id, mode, score));
                }
            } else {
                best = Some((note.note_id, mode, score));
            }
        }
        best.map(|(id, mode, _)| (id, mode))
    }

    fn note_rect(&self, note: &EditableNote, layout: &FallingViewLayout) -> Rect {
        let (x, w) = pitch_xw(layout.rect, note.pitch);
        let lane = self
            .note_lanes
            .get(&note.note_id)
            .copied()
            .unwrap_or(LaneAssignment { index: 0, count: 1 });
        let lane_width = w / lane.count.max(1) as f32;
        let lane_x = x + lane.index as f32 * lane_width;

        let y_start = layout.strike_y
            - ((note.start_tick as f64 - self.playhead_tick) * layout.px_per_tick) as f32;
        let y_end = layout.strike_y
            - ((note.end_tick as f64 - self.playhead_tick) * layout.px_per_tick) as f32;

        let top = y_end.min(y_start);
        let bottom = y_end.max(y_start);
        Rect::from_min_max(
            Pos2::new(lane_x, top),
            Pos2::new((lane_x + lane_width).min(layout.rect.right()), bottom),
        )
    }

    fn draw_falling_area(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let keyboard_height = 112.0;
        let keyboard_rect = Rect::from_min_max(
            Pos2::new(rect.left(), rect.bottom() - keyboard_height),
            Pos2::new(rect.right(), rect.bottom()),
        );
        let fall_rect = Rect::from_min_max(rect.min, Pos2::new(rect.max.x, keyboard_rect.min.y));

        let pitch_min = MIN_PITCH;
        let pitch_max = MAX_PITCH;

        let horizon_ticks = (self.document.ticks_per_beat.max(1) as f64 * 18.0).max(1.0);
        let base_px_per_tick = (fall_rect.height() as f64 / horizon_ticks).max(0.0001);
        let px_per_tick = (base_px_per_tick * self.vertical_zoom as f64).max(0.0001);
        let strike_y = fall_rect.bottom() - 3.0;

        let layout = FallingViewLayout {
            rect: fall_rect,
            strike_y,
            keyboard_rect,
            pitch_min,
            pitch_max,
            horizon_ticks,
            px_per_tick,
        };

        let response = ui.interact(
            fall_rect,
            ui.id().with("falling_view"),
            Sense::click_and_drag(),
        );
        let painter = ui.painter();

        painter.rect_filled(fall_rect, 4.0, Color32::from_rgb(16, 18, 24));
        draw_falling_grid(painter, &layout, self.playhead_tick);

        let pointer_down = ui.input(|i| i.pointer.primary_down());
        if response.drag_started() {
            if let Some(pointer_pos) = response.interact_pointer_pos() {
                let force_resize = ui.input(|i| i.modifiers.shift);
                self.start_drag(pointer_pos, &layout, force_resize);
            }
        } else if self.drag_state.is_some() && pointer_down {
            if let Some(pointer_pos) = ui.input(|i| i.pointer.interact_pos()) {
                self.update_drag(pointer_pos, &layout);
            }
        } else if self.drag_state.is_some() && !pointer_down {
            self.finish_drag();
        } else if response.clicked() {
            if let Some(pointer_pos) = response.interact_pointer_pos() {
                if let Some((note_id, _)) = self.pick_note(pointer_pos, &layout, false) {
                    self.selected_note_id = Some(note_id);
                    if let Some(selected) = self.note_by_id(note_id) {
                        self.audition_pitch(selected.pitch, selected.velocity_on, 160);
                    }
                } else {
                    self.selected_note_id = None;
                }
            }
        }

        if response.secondary_clicked() {
            if let Some(pointer_pos) = response.interact_pointer_pos() {
                if let Some((note_id, _)) = self.pick_note(pointer_pos, &layout, false) {
                    self.selected_note_id = Some(note_id);
                    self.context_menu_note_id = Some(note_id);
                } else {
                    self.context_menu_note_id = None;
                }
            }
        }

        response.context_menu(|ui| {
            if let Some(note_id) = self.context_menu_note_id {
                if ui.button("Delete").clicked() {
                    self.selected_note_id = Some(note_id);
                    self.delete_selected_note();
                    ui.close_menu();
                }
                ui.separator();
                ui.label("Tip: Hold Shift while dragging to resize note length");
                let is_unlocked = self
                    .note_by_id(note_id)
                    .map(|note| note.key_lane_unlocked)
                    .unwrap_or(false);
                let lock_label = if is_unlocked {
                    "Lock Key Lane (This Note)"
                } else {
                    "Unlock Key Lane (This Note)"
                };
                if ui.button(lock_label).clicked() {
                    if let Some(note) = self.note_by_id_mut(note_id) {
                        note.key_lane_unlocked = !note.key_lane_unlocked;
                        self.status_line = if note.key_lane_unlocked {
                            format!("Key lane unlocked for note {}", note_id)
                        } else {
                            format!("Key lane locked for note {}", note_id)
                        };
                    }
                    ui.close_menu();
                }
            } else {
                ui.label("No note selected");
                ui.label("Tip: Hold Shift while dragging to resize note length");
            }
        });

        let mut note_refs: Vec<&EditableNote> = self.document.notes.iter().collect();
        note_refs.sort_by_key(|n| n.start_tick);
        let visible_note_region = Rect::from_min_max(
            Pos2::new(fall_rect.left(), fall_rect.top()),
            Pos2::new(fall_rect.right(), strike_y),
        );
        let selected_onset_tick = self.selected_note_id.and_then(|selected_id| {
            if let Some(preview) = &self.drag_preview {
                if preview.note_id == selected_id {
                    return Some(preview.start_tick);
                }
            }
            self.document
                .notes
                .iter()
                .find(|note| note.note_id == selected_id)
                .map(|note| note.start_tick)
        });

        let onset_connector = selected_onset_tick.and_then(|selected_tick| {
            let total_matches = self
                .document
                .notes
                .iter()
                .filter(|note| {
                    if let Some(preview) = &self.drag_preview {
                        if preview.note_id == note.note_id {
                            return preview.start_tick == selected_tick;
                        }
                    }
                    note.start_tick == selected_tick
                })
                .count();
            if total_matches < 2 {
                return None;
            }

            let onset_y = layout.strike_y
                - ((selected_tick as f64 - self.playhead_tick) * layout.px_per_tick) as f32;
            if onset_y < visible_note_region.top() || onset_y > visible_note_region.bottom() {
                return None;
            }

            let mut span_left = f32::INFINITY;
            let mut span_right = f32::NEG_INFINITY;
            let mut visible_matches = 0usize;

            for note in self.document.notes.iter() {
                let draw_note = if let Some(preview) = &self.drag_preview {
                    if preview.note_id == note.note_id {
                        preview
                    } else {
                        note
                    }
                } else {
                    note
                };
                if draw_note.start_tick != selected_tick {
                    continue;
                }

                let note_rect = self.note_rect(draw_note, &layout);
                if note_rect.bottom() < fall_rect.top() || note_rect.top() > strike_y {
                    continue;
                }
                let clipped_rect = note_rect.intersect(visible_note_region);
                if clipped_rect.width() <= 0.0 || clipped_rect.height() <= 0.0 {
                    continue;
                }

                span_left = span_left.min(clipped_rect.left());
                span_right = span_right.max(clipped_rect.right());
                visible_matches += 1;
            }

            if visible_matches >= 2 && span_left < span_right {
                Some((span_left, span_right, onset_y))
            } else {
                None
            }
        });

        for note in note_refs {
            let draw_note = if let Some(preview) = &self.drag_preview {
                if preview.note_id == note.note_id {
                    preview
                } else {
                    note
                }
            } else {
                note
            };

            let note_rect = self.note_rect(draw_note, &layout);
            if note_rect.bottom() < fall_rect.top() || note_rect.top() > strike_y {
                continue;
            }
            let clipped_rect = note_rect.intersect(visible_note_region);
            if clipped_rect.width() <= 0.0 || clipped_rect.height() <= 0.0 {
                continue;
            }

            let mut color = note_color(
                &self.document.channel_colors,
                draw_note.channel,
                draw_note.pitch,
            );
            if self.selected_note_id == Some(draw_note.note_id) {
                color = Color32::from_rgb(255, 221, 87);
            }

            painter.rect_filled(clipped_rect, 2.0, color);
            painter.rect_stroke(
                clipped_rect,
                2.0,
                Stroke::new(1.0, Color32::from_black_alpha(180)),
            );
        }

        if let Some((span_left, span_right, onset_y)) = onset_connector {
            painter.line_segment(
                [
                    Pos2::new(span_left, onset_y),
                    Pos2::new(span_right, onset_y),
                ],
                Stroke::new(1.0, Color32::from_rgba_unmultiplied(245, 245, 245, 220)),
            );
        }

        painter.line_segment(
            [
                Pos2::new(fall_rect.left(), strike_y),
                Pos2::new(fall_rect.right(), strike_y),
            ],
            Stroke::new(2.0, Color32::from_rgb(220, 220, 220)),
        );

        self.draw_keyboard(ui, &layout);
    }

    fn draw_keyboard(&self, ui: &mut egui::Ui, layout: &FallingViewLayout) {
        let painter = ui.painter();
        painter.rect_filled(layout.keyboard_rect, 3.0, Color32::from_rgb(12, 13, 17));

        let active_channels = self.active_note_channels_at_playhead();
        let white_height = layout.keyboard_rect.height();
        let black_height = white_height * 0.62;

        for pitch in MIN_PITCH..=MAX_PITCH {
            if is_black_key(pitch) {
                continue;
            }
            let rect = piano_key_rect(layout.keyboard_rect, pitch);
            painter.rect_filled(rect, 0.0, Color32::from_rgb(232, 232, 224));
            if let Some(channels) = active_channels.get(&pitch) {
                let band_width = rect.width() / channels.len().max(1) as f32;
                for (index, channel) in channels.iter().enumerate() {
                    let left = rect.left() + index as f32 * band_width;
                    let right = if index + 1 == channels.len() {
                        rect.right()
                    } else {
                        left + band_width
                    };
                    painter.rect_filled(
                        Rect::from_min_max(
                            Pos2::new(left, rect.top()),
                            Pos2::new(right, rect.bottom()),
                        ),
                        0.0,
                        note_color(&self.document.channel_colors, *channel, pitch),
                    );
                }
            }
            painter.rect_stroke(rect, 0.0, Stroke::new(0.8, Color32::from_rgb(50, 50, 50)));
        }

        for pitch in MIN_PITCH..=MAX_PITCH {
            if !is_black_key(pitch) {
                continue;
            }
            let mut rect = piano_key_rect(layout.keyboard_rect, pitch);
            rect.max.y = rect.min.y + black_height;
            painter.rect_filled(rect, 2.0, Color32::from_rgb(28, 28, 32));
            if let Some(channels) = active_channels.get(&pitch) {
                let band_width = rect.width() / channels.len().max(1) as f32;
                for (index, channel) in channels.iter().enumerate() {
                    let left = rect.left() + index as f32 * band_width;
                    let right = if index + 1 == channels.len() {
                        rect.right()
                    } else {
                        left + band_width
                    };
                    painter.rect_filled(
                        Rect::from_min_max(
                            Pos2::new(left, rect.top()),
                            Pos2::new(right, rect.bottom()),
                        ),
                        0.0,
                        note_color(&self.document.channel_colors, *channel, pitch),
                    );
                }
            }
            painter.rect_stroke(rect, 2.0, Stroke::new(1.0, Color32::from_rgb(10, 10, 10)));
        }
    }

    fn handle_close_request(&mut self, ctx: &egui::Context) {
        let close_requested = ctx.input(|i| i.viewport().close_requested());
        if close_requested {
            if self.allow_close_without_prompt_once {
                self.allow_close_without_prompt_once = false;
            } else if self.document.dirty && !self.show_close_prompt {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.show_close_prompt = true;
            }
        }

        if self.show_close_prompt {
            egui::Window::new("Unsaved edits")
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Save touch-up MIDI before closing?");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() {
                            self.save_touchup();
                            self.show_close_prompt = false;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ui.button("Discard").clicked() {
                            if let Ok(mut result) = self.result_state.lock() {
                                *result = EditorResult::cancelled(&self.document.source_path);
                            }
                            self.show_close_prompt = false;
                            self.allow_close_without_prompt_once = true;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_close_prompt = false;
                        }
                    });
                });
        }
    }

    fn update_playback(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame_instant).as_secs_f64();
        self.last_frame_instant = now;

        if self.playing {
            if let Some(audio) = &self.audio_engine {
                self.playhead_tick = self.clamp_tick(audio.playhead_tick());
            } else {
                let now_sec = self.document.tempo_map.tick_to_sec(self.playhead_tick);
                let next_sec = now_sec + (dt * self.speed_factor as f64);
                self.playhead_tick = self.clamp_tick(self.document.tempo_map.sec_to_tick(next_sec));
            }
            if self.playhead_tick >= self.document.max_tick as f64 {
                self.set_playing(false);
            }
            ctx.request_repaint();
        }
    }

    fn step_playhead(&mut self, direction: i32) {
        let delta = self.grid_ticks() as i64 * direction as i64;
        let new_tick = (self.playhead_tick as i64 + delta).max(0) as u64;
        self.set_playhead_tick(new_tick.min(self.document.max_tick) as f64, true);
    }

    fn scrub_playhead_with_wheel(&mut self, scroll_y: f32, fine_mode: bool, smooth_input: bool) {
        if scroll_y.abs() <= f32::EPSILON {
            return;
        }

        let grid_ticks = self.grid_ticks() as f64;
        let direction = if scroll_y > 0.0 { -1.0 } else { 1.0 };
        let delta = if smooth_input {
            // Trackpads emit many tiny smooth deltas, so scale proportionally.
            let ticks_per_scroll_px = if fine_mode {
                (grid_ticks / 120.0).max(0.25)
            } else {
                (grid_ticks / 60.0).max(0.5)
            };
            direction * (scroll_y.abs() as f64) * ticks_per_scroll_px
        } else {
            // Typical wheel notches are near +/-120.
            let ticks_per_wheel_step = if fine_mode {
                (grid_ticks / 4.0).max(1.0)
            } else {
                grid_ticks.max(1.0)
            };
            let wheel_steps = (scroll_y.abs() as f64 / 120.0).max(1.0);
            direction * ticks_per_wheel_step * wheel_steps
        };
        self.set_playhead_tick(self.playhead_tick + delta, true);
    }

    fn zoom_vertical_with_scroll(&mut self, scroll_y: f32, _smooth_input: bool) {
        if scroll_y.abs() <= f32::EPSILON {
            return;
        }
        // One wheel notch (~120) => about 15% zoom change.
        let wheel_steps = scroll_y / 120.0;
        let zoom_factor = 1.15_f32.powf(wheel_steps);
        self.vertical_zoom =
            (self.vertical_zoom * zoom_factor).clamp(MIN_VERTICAL_ZOOM, MAX_VERTICAL_ZOOM);
    }
}

impl App for MidiTouchupApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        if !self.theme_applied {
            apply_dark_theme(ctx.clone());
            self.theme_applied = true;
        }

        self.update_playback(ctx);
        self.handle_close_request(ctx);

        if ctx.input(|i| i.key_pressed(Key::Space)) {
            self.set_playing(!self.playing);
        }
        if ctx.input(|i| i.key_pressed(Key::Delete)) {
            self.delete_selected_note();
        }
        if ctx.input(|i| i.modifiers.command && i.key_pressed(Key::Z)) {
            if ctx.input(|i| i.modifiers.shift) {
                self.redo();
            } else {
                self.undo();
            }
        }
        if ctx.input(|i| i.key_pressed(Key::ArrowLeft)) {
            self.step_playhead(-1);
        }
        if ctx.input(|i| i.key_pressed(Key::ArrowRight)) {
            self.step_playhead(1);
        }

        egui::TopBottomPanel::top("touchup_top_bar").show(ctx, |ui| {
            let metrics = toolbar_metrics(ui.available_width());
            let control_font_size = metrics.font_size;
            let button_height = metrics.button_height;
            let button_width = metrics.button_width;
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 8.0);
            ui.add_space(8.0);

            ui.horizontal_wrapped(|ui| {
                if ui
                    .add_sized(
                        [button_width, button_height],
                        egui::Button::new(RichText::new("Open MIDI").size(control_font_size)),
                    )
                    .clicked()
                {
                    if let Some(path) = FileDialog::new()
                        .add_filter("MIDI", &["mid", "midi"])
                        .set_directory(
                            self.document
                                .source_path
                                .parent()
                                .unwrap_or_else(|| Path::new(".")),
                        )
                        .pick_file()
                    {
                        self.load_new_midi(&path);
                    }
                }

                if ui
                    .add_sized(
                        [button_width * 1.45, button_height],
                        egui::Button::new(
                            RichText::new("Save Touch-Up MIDI").size(control_font_size),
                        ),
                    )
                    .clicked()
                {
                    self.save_touchup();
                }

                let snap_label = if self.snap_enabled {
                    "Snap: ON"
                } else {
                    "Snap: OFF"
                };
                if ui
                    .add_sized(
                        [button_width, button_height],
                        egui::Button::new(RichText::new(snap_label).size(control_font_size)),
                    )
                    .clicked()
                {
                    self.snap_enabled = !self.snap_enabled;
                }

                if self.snap_enabled {
                    egui::ComboBox::from_label(
                        RichText::new("Snap To").size(control_font_size - 2.0),
                    )
                    .selected_text(RichText::new(self.snap_target.label()).size(control_font_size))
                    .width(180.0)
                    .show_ui(ui, |ui| {
                        for option in SnapTarget::all() {
                            ui.selectable_value(
                                &mut self.snap_target,
                                option,
                                RichText::new(option.label()).size(control_font_size - 1.0),
                            );
                        }
                    });
                }

                egui::ComboBox::from_label(RichText::new("Grid").size(control_font_size - 2.0))
                    .selected_text(RichText::new(self.grid_option.label()).size(control_font_size))
                    .width(130.0)
                    .show_ui(ui, |ui| {
                        for option in GridOption::all() {
                            ui.selectable_value(
                                &mut self.grid_option,
                                option,
                                RichText::new(option.label()).size(control_font_size - 1.0),
                            );
                        }
                    });

                let play_pause_icon = if self.playing { "⏸" } else { "▶" };
                if ui
                    .add_sized(
                        [button_width * 0.6, button_height],
                        egui::Button::new(
                            RichText::new(play_pause_icon).size(control_font_size + 6.0),
                        ),
                    )
                    .on_hover_text(if self.playing { "Pause" } else { "Play" })
                    .clicked()
                {
                    self.set_playing(!self.playing);
                }
                if ui
                    .add_sized(
                        [button_width * 0.8, button_height],
                        egui::Button::new(RichText::new("Step -").size(control_font_size)),
                    )
                    .clicked()
                {
                    self.step_playhead(-1);
                }
                if ui
                    .add_sized(
                        [button_width * 0.8, button_height],
                        egui::Button::new(RichText::new("Step +").size(control_font_size)),
                    )
                    .clicked()
                {
                    self.step_playhead(1);
                }

                egui::ComboBox::from_label(RichText::new("Speed").size(control_font_size - 2.0))
                    .selected_text(RichText::new(&self.speed_label).size(control_font_size))
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        for (label, factor) in [
                            ("0.5x", 0.5_f32),
                            ("1.0x", 1.0_f32),
                            ("1.25x", 1.25_f32),
                            ("1.5x", 1.5_f32),
                            ("2.0x", 2.0_f32),
                        ] {
                            if ui
                                .selectable_label(
                                    self.speed_label == label,
                                    RichText::new(label).size(control_font_size - 1.0),
                                )
                                .clicked()
                            {
                                self.set_speed(factor, label.to_string());
                            }
                        }
                    });

                if ui
                    .add_sized(
                        [button_width, button_height],
                        egui::Button::new(
                            RichText::new(if self.audio_muted { "Unmute" } else { "Mute" })
                                .size(control_font_size),
                        ),
                    )
                    .clicked()
                {
                    self.set_audio_mute(!self.audio_muted);
                }

                ui.vertical(|ui| {
                    ui.label(RichText::new("Volume").size(control_font_size - 8.0));
                    let mut volume = self.audio_volume;
                    let response = ui.add_sized(
                        [button_width * 1.1, button_height - 8.0],
                        egui::Slider::new(&mut volume, 0.0..=1.0).show_value(false),
                    );
                    if response.changed() {
                        self.set_audio_volume(volume);
                    }
                });

                ui.horizontal(|ui| {
                    ui.label(RichText::new("Octave").size(control_font_size - 2.0));
                    if ui
                        .add_sized(
                            [button_width * 0.32, button_height],
                            egui::Button::new(RichText::new("-").size(control_font_size)),
                        )
                        .on_hover_text("Shift every note down one octave")
                        .clicked()
                    {
                        self.request_octave_shift(-1);
                    }
                    ui.add_sized(
                        [44.0, button_height],
                        egui::Label::new(
                            RichText::new(octave_offset_label(self.octave_offset))
                                .size(control_font_size),
                        ),
                    );
                    if ui
                        .add_sized(
                            [button_width * 0.32, button_height],
                            egui::Button::new(RichText::new("+").size(control_font_size)),
                        )
                        .on_hover_text("Shift every note up one octave")
                        .clicked()
                    {
                        self.request_octave_shift(1);
                    }
                });

                if ui
                    .add_enabled(
                        !self.undo_stack.is_empty(),
                        egui::Button::new(RichText::new("Undo").size(control_font_size))
                            .min_size(Vec2::new(button_width * 0.8, button_height)),
                    )
                    .on_hover_text("Undo the last edit")
                    .clicked()
                {
                    self.undo();
                }
                if ui
                    .add_enabled(
                        !self.redo_stack.is_empty(),
                        egui::Button::new(RichText::new("Redo").size(control_font_size))
                            .min_size(Vec2::new(button_width * 0.8, button_height)),
                    )
                    .on_hover_text("Redo the last undone edit")
                    .clicked()
                {
                    self.redo();
                }
            });

            ui.add_space(6.0);
            egui::Frame::none()
                .fill(Color32::from_rgb(22, 30, 44))
                .stroke(Stroke::new(1.5, Color32::from_rgb(92, 116, 155)))
                .rounding(egui::Rounding::same(8.0))
                .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Timeline").size(control_font_size).strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                RichText::new(format!("{:.2}s", self.playback_seconds()))
                                    .size(control_font_size - 1.0),
                            );
                        });
                    });

                    let (timeline_rect, timeline_response) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), 44.0),
                        Sense::click_and_drag(),
                    );
                    let timeline_painter = ui.painter();

                    timeline_painter.rect_filled(timeline_rect, 6.0, Color32::from_rgb(38, 52, 74));

                    let max_tick_f64 = self.document.max_tick.max(1) as f64;
                    let progress_ratio = (self.playhead_tick / max_tick_f64).clamp(0.0, 1.0) as f32;
                    let progress_x =
                        timeline_rect.left() + (timeline_rect.width() * progress_ratio);
                    let progress_rect = Rect::from_min_max(
                        timeline_rect.min,
                        Pos2::new(progress_x, timeline_rect.max.y),
                    );
                    timeline_painter.rect_filled(
                        progress_rect,
                        6.0,
                        Color32::from_rgb(107, 153, 224),
                    );
                    timeline_painter.line_segment(
                        [
                            Pos2::new(progress_x, timeline_rect.top()),
                            Pos2::new(progress_x, timeline_rect.bottom()),
                        ],
                        Stroke::new(2.0, Color32::from_rgb(250, 250, 252)),
                    );

                    if timeline_response.drag_started() || timeline_response.clicked() {
                        if let Some(pointer_pos) = timeline_response.interact_pointer_pos() {
                            self.begin_timeline_scrub_audio_mute();
                            let fine_mode = ui.input(|i| i.modifiers.shift);
                            self.timeline_drag_state = Some(TimelineDragState {
                                start_x: pointer_pos.x,
                                start_tick: self.playhead_tick,
                                fine_mode,
                            });
                            if !fine_mode {
                                self.set_playhead_tick(
                                    self.tick_from_timeline_x(timeline_rect, pointer_pos.x),
                                    true,
                                );
                            }
                        }
                    }

                    if timeline_response.dragged() {
                        if let Some(pointer_pos) = timeline_response.interact_pointer_pos() {
                            self.begin_timeline_scrub_audio_mute();
                            let fine_mode_held = ui.input(|i| i.modifiers.shift);
                            if fine_mode_held {
                                if self.timeline_drag_state.is_none() {
                                    self.timeline_drag_state = Some(TimelineDragState {
                                        start_x: pointer_pos.x,
                                        start_tick: self.playhead_tick,
                                        fine_mode: true,
                                    });
                                }
                                if let Some(state) = &self.timeline_drag_state {
                                    let fine_ticks_per_px =
                                        (self.grid_ticks() as f64 / 8.0).max(1.0);
                                    let delta_tick =
                                        (pointer_pos.x - state.start_x) as f64 * fine_ticks_per_px;
                                    let drag_base_tick = if state.fine_mode {
                                        state.start_tick
                                    } else {
                                        self.playhead_tick
                                    };
                                    self.set_playhead_tick(drag_base_tick + delta_tick, true);
                                }
                            } else {
                                self.set_playhead_tick(
                                    self.tick_from_timeline_x(timeline_rect, pointer_pos.x),
                                    true,
                                );
                            }
                        }
                    }

                    if !ui.input(|i| i.pointer.primary_down()) {
                        self.timeline_drag_state = None;
                        self.end_timeline_scrub_audio_mute();
                    }
                });

            ui.add_space(6.0);
        });

        egui::TopBottomPanel::bottom("touchup_status_bar")
            .exact_height(34.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&self.status_line).size(14.0).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!(
                                "Source: {}",
                                self.document
                                    .source_path
                                    .file_name()
                                    .map(|name| name.to_string_lossy().to_string())
                                    .unwrap_or_else(|| self
                                        .document
                                        .source_path
                                        .display()
                                        .to_string())
                            ))
                            .size(13.0),
                        );
                    });
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_rect_before_wrap();
            let pointer_over_canvas = ui.rect_contains_pointer(available)
                && pointer_targets_canvas(ui.input(|input| input.pointer.hover_pos()), available);
            if pointer_over_canvas {
                let (smooth_scroll_y, raw_scroll_y, shift_down, zoom_modifier_down) =
                    ui.input(|input| {
                        (
                            input.smooth_scroll_delta.y,
                            input.raw_scroll_delta.y,
                            input.modifiers.shift,
                            input.modifiers.ctrl || input.modifiers.command,
                        )
                    });
                let (scroll_y, smooth_input) = if smooth_scroll_y.abs() > f32::EPSILON {
                    (smooth_scroll_y, true)
                } else {
                    (raw_scroll_y, false)
                };
                if scroll_y.abs() > f32::EPSILON {
                    if zoom_modifier_down {
                        self.zoom_vertical_with_scroll(scroll_y, smooth_input);
                    } else {
                        self.scrub_playhead_with_wheel(scroll_y, shift_down, smooth_input);
                    }
                }
            }
            self.draw_falling_area(ui, available);
        });
    }
}

impl AudioEngineHandle {
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    fn new(sf2_override: Option<PathBuf>, document: &MidiDocument) -> Result<Self, String> {
        let sf2_path = resolve_soundfont_path(sf2_override)?;
        let sf2_file = fs::File::open(&sf2_path)
            .map_err(|err| format!("Failed to open SoundFont {}: {err}", sf2_path.display()))?;
        let mut sf2_reader = BufReader::new(sf2_file);
        let sound_font =
            Arc::new(SoundFont::new(&mut sf2_reader).map_err(|err| {
                format!("Failed to parse SoundFont {}: {err}", sf2_path.display())
            })?);

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "No default audio output device available".to_string())?;
        let supported = device
            .default_output_config()
            .map_err(|err| format!("Failed to query output device config: {err}"))?;
        let sample_format = supported.sample_format();
        let stream_config = supported.config();
        let channels = stream_config.channels as usize;
        let sample_rate = stream_config.sample_rate.0 as f64;

        let mut settings = SynthesizerSettings::new(stream_config.sample_rate.0 as i32);
        settings.enable_reverb_and_chorus = false;
        let mut synth = Synthesizer::new(&sound_font, &settings)
            .map_err(|err| format!("Failed to create synthesizer: {err}"))?;
        // Force piano for all channels in v1.
        for channel in 0..16 {
            synth.process_midi_message(channel, 0xC0, 0, 0);
        }

        let song = SongRenderData::from_document(document);
        let runtime = AudioRuntime {
            synth,
            song,
            event_index: 0,
            playback_sec: 0.0,
            playing: false,
            speed: 1.0,
            muted: false,
            volume: 1.0,
            preview_voices: Vec::new(),
            sample_rate,
        };

        let (sender, receiver) = unbounded::<AudioCommand>();
        let telemetry = Arc::new(AudioTelemetry::default());
        telemetry.set_playhead_tick(0.0);
        telemetry.set_meter(0.0);
        let runtime = Arc::new(Mutex::new(runtime));

        let stream = match sample_format {
            cpal::SampleFormat::F32 => {
                let receiver = receiver.clone();
                let telemetry_cb = telemetry.clone();
                let runtime_cb = runtime.clone();
                let err_sf2_path = sf2_path.clone();
                device
                    .build_output_stream(
                        &stream_config,
                        move |data: &mut [f32], _info| {
                            if let Ok(mut runtime) = runtime_cb.lock() {
                                render_audio_chunk_f32(
                                    data,
                                    channels,
                                    &receiver,
                                    &telemetry_cb,
                                    &mut runtime,
                                );
                            } else {
                                for sample in data.iter_mut() {
                                    *sample = 0.0;
                                }
                            }
                        },
                        move |err| {
                            eprintln!(
                                "[touchup-audio] stream error for {}: {err}",
                                err_sf2_path.display()
                            );
                        },
                        None,
                    )
                    .map_err(|err| format!("Failed building f32 output stream: {err}"))?
            }
            cpal::SampleFormat::I16 => {
                let receiver = receiver.clone();
                let telemetry_cb = telemetry.clone();
                let runtime_cb = runtime.clone();
                let err_sf2_path = sf2_path.clone();
                device
                    .build_output_stream(
                        &stream_config,
                        move |data: &mut [i16], _info| {
                            if let Ok(mut runtime) = runtime_cb.lock() {
                                render_audio_chunk_i16(
                                    data,
                                    channels,
                                    &receiver,
                                    &telemetry_cb,
                                    &mut runtime,
                                );
                            } else {
                                for sample in data.iter_mut() {
                                    *sample = 0;
                                }
                            }
                        },
                        move |err| {
                            eprintln!(
                                "[touchup-audio] stream error for {}: {err}",
                                err_sf2_path.display()
                            );
                        },
                        None,
                    )
                    .map_err(|err| format!("Failed building i16 output stream: {err}"))?
            }
            cpal::SampleFormat::U16 => {
                let receiver = receiver.clone();
                let telemetry_cb = telemetry.clone();
                let runtime_cb = runtime.clone();
                let err_sf2_path = sf2_path.clone();
                device
                    .build_output_stream(
                        &stream_config,
                        move |data: &mut [u16], _info| {
                            if let Ok(mut runtime) = runtime_cb.lock() {
                                render_audio_chunk_u16(
                                    data,
                                    channels,
                                    &receiver,
                                    &telemetry_cb,
                                    &mut runtime,
                                );
                            } else {
                                for sample in data.iter_mut() {
                                    *sample = u16::MAX / 2;
                                }
                            }
                        },
                        move |err| {
                            eprintln!(
                                "[touchup-audio] stream error for {}: {err}",
                                err_sf2_path.display()
                            );
                        },
                        None,
                    )
                    .map_err(|err| format!("Failed building u16 output stream: {err}"))?
            }
            other => {
                return Err(format!("Unsupported output sample format: {other:?}"));
            }
        };
        stream
            .play()
            .map_err(|err| format!("Failed to start audio stream: {err}"))?;

        Ok(Self {
            sender,
            telemetry,
            stream,
        })
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    fn new(_sf2_override: Option<PathBuf>, _document: &MidiDocument) -> Result<Self, String> {
        Err(
            "Audio playback is currently enabled on Windows/macOS builds. Running muted fallback on this platform."
                .to_string(),
        )
    }

    fn send(&self, command: AudioCommand) {
        let _ = self.sender.send(command);
    }

    fn playhead_tick(&self) -> f64 {
        self.telemetry.playhead_tick()
    }
}

impl Drop for AudioEngineHandle {
    fn drop(&mut self) {
        let _ = self.sender.send(AudioCommand::Shutdown);
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        let _ = self.stream.pause();
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn resolve_soundfont_path(sf2_override: Option<PathBuf>) -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = sf2_override {
        if path.is_file() {
            return Ok(path);
        }
        candidates.push(path);
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(
                exe_dir
                    .join("assets")
                    .join("soundfonts")
                    .join(DEFAULT_SF2_FILENAME),
            );
            candidates.push(
                exe_dir
                    .join("..")
                    .join("assets")
                    .join("soundfonts")
                    .join(DEFAULT_SF2_FILENAME),
            );
        }
    }

    candidates.push(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("soundfonts")
            .join(DEFAULT_SF2_FILENAME),
    );

    for candidate in candidates.iter() {
        if candidate.is_file() {
            return Ok(candidate.to_path_buf());
        }
    }

    let checked = candidates
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    Err(format!(
        "SoundFont not found. Use --sf2 <path> or place {DEFAULT_SF2_FILENAME} at one of:\n{checked}"
    ))
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn all_notes_off(synth: &mut Synthesizer) {
    synth.note_off_all(true);
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn retrigger_active_notes(runtime: &mut AudioRuntime, tick: u64) {
    for (channel, pitch, velocity) in runtime.song.active_notes_at_tick(tick) {
        runtime
            .synth
            .note_on(channel as i32, pitch as i32, velocity as i32);
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn apply_song_event(runtime: &mut AudioRuntime, event: &SongEvent) {
    match event.kind {
        SongEventKind::NoteOn {
            channel,
            pitch,
            velocity,
        } => runtime
            .synth
            .note_on(channel as i32, pitch as i32, velocity as i32),
        SongEventKind::NoteOff { channel, pitch } => {
            runtime.synth.note_off(channel as i32, pitch as i32)
        }
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn seek_runtime_to_tick(runtime: &mut AudioRuntime, tick: u64) {
    let clamped_tick = tick.min(runtime.song.max_tick);
    all_notes_off(&mut runtime.synth);
    runtime.preview_voices.clear();
    runtime.playback_sec = runtime.song.tempo_map.tick_to_sec(clamped_tick as f64);
    runtime.event_index = runtime.song.event_index_after_tick(clamped_tick);
    if runtime.playing {
        retrigger_active_notes(runtime, clamped_tick);
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn process_audio_commands(runtime: &mut AudioRuntime, receiver: &Receiver<AudioCommand>) -> bool {
    let mut shutdown_requested = false;
    while let Ok(cmd) = receiver.try_recv() {
        match cmd {
            AudioCommand::Play => {
                runtime.playing = true;
                let tick = runtime
                    .song
                    .tempo_map
                    .sec_to_tick(runtime.playback_sec)
                    .max(0.0) as u64;
                seek_runtime_to_tick(runtime, tick);
            }
            AudioCommand::Pause => {
                runtime.playing = false;
                all_notes_off(&mut runtime.synth);
            }
            AudioCommand::SeekTick(tick) => {
                seek_runtime_to_tick(runtime, tick);
            }
            AudioCommand::SetSpeed(speed) => {
                runtime.speed = speed.max(0.1);
            }
            AudioCommand::SetMute(muted) => {
                runtime.muted = muted;
            }
            AudioCommand::SetVolume(volume) => {
                runtime.volume = volume.clamp(0.0, 1.0);
            }
            AudioCommand::PreviewNote {
                pitch,
                velocity,
                duration_ms,
            } => {
                let channel = 15_u8;
                runtime
                    .synth
                    .note_on(channel as i32, pitch as i32, velocity.max(1) as i32);
                let samples = ((duration_ms.max(20) as f64 / 1000.0) * runtime.sample_rate)
                    .round()
                    .max(1.0) as u32;
                runtime.preview_voices.push(PreviewVoice {
                    channel,
                    pitch,
                    remaining_samples: samples,
                });
            }
            AudioCommand::LoadSong(song) => {
                runtime.song = song;
                runtime.playing = false;
                seek_runtime_to_tick(runtime, 0);
            }
            AudioCommand::Shutdown => {
                runtime.playing = false;
                all_notes_off(&mut runtime.synth);
                shutdown_requested = true;
            }
        }
    }
    shutdown_requested
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn advance_preview_voices(runtime: &mut AudioRuntime) {
    for idx in (0..runtime.preview_voices.len()).rev() {
        if runtime.preview_voices[idx].remaining_samples == 0 {
            let preview = runtime.preview_voices.remove(idx);
            runtime
                .synth
                .note_off(preview.channel as i32, preview.pitch as i32);
            continue;
        }
        runtime.preview_voices[idx].remaining_samples = runtime.preview_voices[idx]
            .remaining_samples
            .saturating_sub(1);
        if runtime.preview_voices[idx].remaining_samples == 0 {
            let preview = runtime.preview_voices.remove(idx);
            runtime
                .synth
                .note_off(preview.channel as i32, preview.pitch as i32);
        }
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn render_one_sample(runtime: &mut AudioRuntime) -> (f32, f32) {
    if runtime.playing {
        runtime.playback_sec += runtime.speed as f64 / runtime.sample_rate;
        while runtime.event_index < runtime.song.events.len()
            && runtime.song.events[runtime.event_index].sec <= runtime.playback_sec + 1e-9
        {
            let event = runtime.song.events[runtime.event_index].clone();
            apply_song_event(runtime, &event);
            runtime.event_index += 1;
        }
        if runtime.playback_sec >= runtime.song.max_sec {
            runtime.playback_sec = runtime.song.max_sec;
            runtime.playing = false;
            all_notes_off(&mut runtime.synth);
        }
    }

    advance_preview_voices(runtime);

    let mut left = [0.0_f32; 1];
    let mut right = [0.0_f32; 1];
    runtime.synth.render(&mut left, &mut right);

    let gain = if runtime.muted { 0.0 } else { runtime.volume };
    (left[0] * gain, right[0] * gain)
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn render_audio_chunk_f32(
    data: &mut [f32],
    channels: usize,
    receiver: &Receiver<AudioCommand>,
    telemetry: &AudioTelemetry,
    runtime: &mut AudioRuntime,
) {
    let shutdown = process_audio_commands(runtime, receiver);
    if shutdown {
        for sample in data.iter_mut() {
            *sample = 0.0;
        }
        telemetry.set_meter(0.0);
        return;
    }

    let mut meter = 0.0_f32;
    for frame in data.chunks_mut(channels.max(1)) {
        let (left, right) = render_one_sample(runtime);
        meter = meter.max(left.abs().max(right.abs()));
        for (ch_idx, slot) in frame.iter_mut().enumerate() {
            *slot = if ch_idx % 2 == 0 { left } else { right };
        }
    }
    let tick = runtime
        .song
        .tempo_map
        .sec_to_tick(runtime.playback_sec)
        .clamp(0.0, runtime.song.max_tick as f64);
    telemetry.set_playhead_tick(tick);
    telemetry.set_meter(meter);
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn render_audio_chunk_i16(
    data: &mut [i16],
    channels: usize,
    receiver: &Receiver<AudioCommand>,
    telemetry: &AudioTelemetry,
    runtime: &mut AudioRuntime,
) {
    let mut scratch = vec![0.0_f32; data.len()];
    render_audio_chunk_f32(&mut scratch, channels, receiver, telemetry, runtime);
    for (dst, sample) in data.iter_mut().zip(scratch.iter()) {
        *dst = (*sample * i16::MAX as f32)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn render_audio_chunk_u16(
    data: &mut [u16],
    channels: usize,
    receiver: &Receiver<AudioCommand>,
    telemetry: &AudioTelemetry,
    runtime: &mut AudioRuntime,
) {
    let mut scratch = vec![0.0_f32; data.len()];
    render_audio_chunk_f32(&mut scratch, channels, receiver, telemetry, runtime);
    for (dst, sample) in data.iter_mut().zip(scratch.iter()) {
        let normalized = ((*sample + 1.0) * 0.5).clamp(0.0, 1.0);
        *dst = (normalized * u16::MAX as f32)
            .round()
            .clamp(0.0, u16::MAX as f32) as u16;
    }
}

fn apply_dark_theme(ctx: egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.visuals = egui::Visuals::dark();
    style.visuals.window_fill = Color32::from_rgb(15, 18, 24);
    style.visuals.panel_fill = Color32::from_rgb(15, 18, 24);
    style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(24, 28, 36);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(36, 42, 56);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(52, 64, 82);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(72, 88, 116);
    ctx.set_style(style);
}

fn is_black_key(pitch: u8) -> bool {
    matches!(pitch % 12, 1 | 3 | 6 | 8 | 10)
}

fn white_key_count() -> usize {
    (MIN_PITCH..=MAX_PITCH)
        .filter(|pitch| !is_black_key(*pitch))
        .count()
}

fn white_index_before_pitch(pitch: u8) -> usize {
    (MIN_PITCH..pitch)
        .filter(|candidate| !is_black_key(*candidate))
        .count()
}

fn pitch_xw(rect: Rect, pitch: u8) -> (f32, f32) {
    let white_count = white_key_count() as f32;
    let white_width = if white_count > 0.0 {
        rect.width() / white_count
    } else {
        1.0
    };

    if is_black_key(pitch) {
        // Black keys sit between adjacent white keys and are narrower.
        let prev_white = pitch.saturating_sub(1);
        let prev_white_idx = white_index_before_pitch(prev_white) as f32;
        let boundary_x = rect.left() + (prev_white_idx + 1.0) * white_width;
        let width = (white_width * 0.62).max(3.0);
        let max_left = (rect.right() - width).max(rect.left());
        let x = (boundary_x - width * 0.5).clamp(rect.left(), max_left);
        (x, width)
    } else {
        let white_idx = white_index_before_pitch(pitch) as f32;
        let x = rect.left() + white_idx * white_width;
        (x, white_width.max(3.0))
    }
}

fn pitch_from_x(layout: &FallingViewLayout, x: f32) -> u8 {
    let mut best_pitch = layout.pitch_min;
    let mut best_distance = f32::MAX;
    let clamped_x = x.clamp(layout.rect.left(), layout.rect.right());

    for pitch in layout.pitch_min..=layout.pitch_max {
        let (key_x, key_w) = pitch_xw(layout.rect, pitch);
        let center = key_x + (key_w * 0.5);
        let distance = (clamped_x - center).abs();
        if distance < best_distance {
            best_distance = distance;
            best_pitch = pitch;
        }
    }
    best_pitch
}

fn piano_key_rect(keyboard_rect: Rect, pitch: u8) -> Rect {
    let (x, w) = pitch_xw(keyboard_rect, pitch);
    Rect::from_min_size(
        Pos2::new(x, keyboard_rect.top()),
        Vec2::new(w, keyboard_rect.height()),
    )
}

fn draw_falling_grid(painter: &egui::Painter, layout: &FallingViewLayout, playhead_tick: f64) {
    let white_stroke = Stroke::new(0.5, Color32::from_rgba_unmultiplied(60, 64, 74, 80));
    for pitch in layout.pitch_min..=layout.pitch_max {
        if is_black_key(pitch) {
            continue;
        }
        let (x, _width) = pitch_xw(layout.rect, pitch);
        let stroke = if pitch % 12 == 0 {
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(112, 126, 150, 110))
        } else {
            white_stroke
        };
        painter.line_segment(
            [
                Pos2::new(x, layout.rect.top()),
                Pos2::new(x, layout.rect.bottom()),
            ],
            stroke,
        );
    }

    if let Some(last_white) = (layout.pitch_min..=layout.pitch_max)
        .rev()
        .find(|pitch| !is_black_key(*pitch))
    {
        let (x, width) = pitch_xw(layout.rect, last_white);
        let right_x = (x + width).min(layout.rect.right());
        painter.line_segment(
            [
                Pos2::new(right_x, layout.rect.top()),
                Pos2::new(right_x, layout.rect.bottom()),
            ],
            white_stroke,
        );
    }

    let line_step = layout.horizon_ticks / 8.0;
    for i in 0..=8 {
        let tick = playhead_tick + (i as f64 * line_step);
        let y = layout.strike_y - ((tick - playhead_tick) * layout.px_per_tick) as f32;
        painter.line_segment(
            [
                Pos2::new(layout.rect.left(), y),
                Pos2::new(layout.rect.right(), y),
            ],
            Stroke::new(0.7, Color32::from_rgba_unmultiplied(70, 70, 78, 120)),
        );
    }

    painter.text(
        Pos2::new(layout.rect.left() + 8.0, layout.rect.top() + 8.0),
        Align2::LEFT_TOP,
        "Falling Bars Touch-Up",
        FontId::proportional(14.0),
        Color32::from_rgb(190, 196, 210),
    );
}

fn parse_header_bytes(bytes: &[u8]) -> Result<(u16, u16), String> {
    if bytes.len() < 14 {
        return Err("MIDI file too short for header".to_string());
    }
    if &bytes[0..4] != b"MThd" {
        return Err("Invalid MIDI header chunk signature".to_string());
    }
    let header_len = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if header_len != 6 {
        return Err(format!("Unsupported MIDI header length: {header_len}"));
    }
    let format_u16 = u16::from_be_bytes([bytes[8], bytes[9]]);
    let division_u16 = u16::from_be_bytes([bytes[12], bytes[13]]);
    Ok((format_u16, division_u16))
}

fn load_midi_document(path: &Path) -> Result<MidiDocument, String> {
    let bytes = fs::read(path)
        .map_err(|err| format!("Failed to read MIDI file {}: {err}", path.display()))?;
    let (format_u16, division_u16) = parse_header_bytes(&bytes)?;

    let smf = Smf::parse(&bytes).map_err(|err| format!("MIDI parse failed: {err}"))?;

    let ticks_per_beat = match smf.header.timing {
        midly::Timing::Metrical(tpq) => tpq.as_int(),
        midly::Timing::Timecode(_fps, _subframe) => 480,
    };

    let mut notes = Vec::new();
    let mut preserved_tracks = Vec::new();
    let mut next_note_id: u64 = 1;
    let mut max_tick = 0_u64;
    let mut tempo_us_per_beat = DEFAULT_TEMPO_US_PER_BEAT;
    let mut tempo_events: Vec<TempoEvent> = Vec::new();
    let mut channel_colors = ChannelColorMap::new();

    for (track_index, track) in smf.tracks.iter().enumerate() {
        let mut abs_tick: u64 = 0;
        let mut order: u32 = 0;
        let mut preserved: Vec<PreservedEvent> = Vec::new();
        let mut active: HashMap<(u8, u8), Vec<(u64, u8)>> = HashMap::new();

        for event in track.iter() {
            abs_tick = abs_tick.saturating_add(event.delta.as_int() as u64);
            max_tick = max_tick.max(abs_tick);

            match event.kind {
                TrackEventKind::Midi { channel, message } => {
                    let ch = channel.as_int();
                    match message {
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            active
                                .entry((ch, key.as_int()))
                                .or_default()
                                .push((abs_tick, vel.as_int()));
                        }
                        MidiMessage::NoteOn { key, vel } if vel.as_int() == 0 => {
                            let key_num = key.as_int();
                            if let Some(starts) = active.get_mut(&(ch, key_num)) {
                                if let Some((start_tick, vel_on)) = starts.pop() {
                                    notes.push(EditableNote {
                                        note_id: next_note_id,
                                        track_index,
                                        channel: ch,
                                        pitch: key_num,
                                        start_tick,
                                        end_tick: abs_tick.max(start_tick + 1),
                                        velocity_on: vel_on,
                                        velocity_off: 0,
                                        key_lane_unlocked: false,
                                    });
                                    next_note_id += 1;
                                }
                            }
                        }
                        MidiMessage::NoteOff { key, vel } => {
                            let key_num = key.as_int();
                            if let Some(starts) = active.get_mut(&(ch, key_num)) {
                                if let Some((start_tick, vel_on)) = starts.pop() {
                                    notes.push(EditableNote {
                                        note_id: next_note_id,
                                        track_index,
                                        channel: ch,
                                        pitch: key_num,
                                        start_tick,
                                        end_tick: abs_tick.max(start_tick + 1),
                                        velocity_on: vel_on,
                                        velocity_off: vel.as_int(),
                                        key_lane_unlocked: false,
                                    });
                                    next_note_id += 1;
                                }
                            }
                        }
                        other => {
                            preserved.push(PreservedEvent {
                                tick: abs_tick,
                                order,
                                raw_bytes: encode_midi_message(ch, &other),
                                is_end_of_track: false,
                            });
                        }
                    }
                }
                TrackEventKind::Meta(meta) => {
                    if abs_tick == 0 {
                        if let MetaMessage::Text(bytes) = meta {
                            if let Some(parsed) = parse_color_map_text(bytes) {
                                channel_colors = parsed;
                            }
                        }
                    }
                    if let MetaMessage::Tempo(value) = meta {
                        let tempo_value = value.as_int().max(1);
                        if tempo_events.is_empty() {
                            tempo_us_per_beat = tempo_value;
                        }
                        tempo_events.push(TempoEvent {
                            tick: abs_tick,
                            us_per_beat: tempo_value,
                        });
                    }
                    preserved.push(PreservedEvent {
                        tick: abs_tick,
                        order,
                        raw_bytes: encode_meta_message(&meta),
                        is_end_of_track: matches!(meta, MetaMessage::EndOfTrack),
                    });
                }
                TrackEventKind::SysEx(bytes) => {
                    preserved.push(PreservedEvent {
                        tick: abs_tick,
                        order,
                        raw_bytes: encode_sysex(bytes),
                        is_end_of_track: false,
                    });
                }
                TrackEventKind::Escape(bytes) => {
                    preserved.push(PreservedEvent {
                        tick: abs_tick,
                        order,
                        raw_bytes: encode_escape(bytes),
                        is_end_of_track: false,
                    });
                }
            }
            order = order.saturating_add(1);
        }

        for ((channel, pitch), starts) in active.into_iter() {
            for (start_tick, vel_on) in starts.into_iter() {
                let fallback_end = start_tick + (ticks_per_beat as u64 / 4).max(1);
                notes.push(EditableNote {
                    note_id: next_note_id,
                    track_index,
                    channel,
                    pitch,
                    start_tick,
                    end_tick: fallback_end,
                    velocity_on: vel_on,
                    velocity_off: 64,
                    key_lane_unlocked: false,
                });
                next_note_id += 1;
                max_tick = max_tick.max(fallback_end);
            }
        }

        preserved_tracks.push(preserved);
    }

    notes.sort_by_key(|n| (n.start_tick, n.pitch, n.track_index));
    let tempo_map = if tempo_events.is_empty() {
        TempoMap::default(ticks_per_beat)
    } else {
        TempoMap::from_events(ticks_per_beat, DEFAULT_TEMPO_US_PER_BEAT, tempo_events)
    };
    let effective_tempo = tempo_map
        .events
        .first()
        .map(|e| e.us_per_beat)
        .unwrap_or(tempo_us_per_beat.max(1));

    Ok(MidiDocument {
        source_path: path.to_path_buf(),
        format_u16,
        division_u16,
        ticks_per_beat,
        tempo_us_per_beat: effective_tempo,
        tempo_map,
        notes,
        channel_colors,
        preserved_tracks,
        max_tick: max_tick.max(ticks_per_beat as u64 * 8),
        next_note_id,
        dirty: false,
    })
}

fn save_midi_document(doc: &MidiDocument) -> Result<PathBuf, String> {
    let output_path = next_touchup_path(&doc.source_path);
    let track_count = doc.preserved_tracks.len();

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"MThd");
    out.extend_from_slice(&6_u32.to_be_bytes());
    out.extend_from_slice(&doc.format_u16.to_be_bytes());
    out.extend_from_slice(&(track_count as u16).to_be_bytes());
    out.extend_from_slice(&doc.division_u16.to_be_bytes());

    let mut notes_per_track: Vec<Vec<&EditableNote>> = vec![Vec::new(); track_count];
    for note in doc.notes.iter() {
        if note.track_index < track_count {
            notes_per_track[note.track_index].push(note);
        }
    }

    for track_idx in 0..track_count {
        let mut events: Vec<RawEvent> = Vec::new();

        for preserved in doc.preserved_tracks[track_idx].iter() {
            if preserved.is_end_of_track {
                continue;
            }
            events.push(RawEvent {
                tick: preserved.tick,
                order: preserved.order as u64,
                raw_bytes: preserved.raw_bytes.clone(),
            });
        }

        let mut note_order_seed: u64 = 1_000_000;
        for note in notes_per_track[track_idx].iter() {
            let note_off = RawEvent {
                tick: note.end_tick,
                order: note_order_seed,
                raw_bytes: vec![0x80 | (note.channel & 0x0F), note.pitch, note.velocity_off],
            };
            note_order_seed += 1;
            let note_on = RawEvent {
                tick: note.start_tick,
                order: note_order_seed,
                raw_bytes: vec![
                    0x90 | (note.channel & 0x0F),
                    note.pitch,
                    note.velocity_on.max(1),
                ],
            };
            note_order_seed += 1;
            events.push(note_off);
            events.push(note_on);
        }

        let end_tick = events.iter().map(|ev| ev.tick).max().unwrap_or(0);
        events.push(RawEvent {
            tick: end_tick,
            order: u64::MAX - 1,
            raw_bytes: vec![0xFF, 0x2F, 0x00],
        });

        events.sort_by(|a, b| (a.tick, a.order).cmp(&(b.tick, b.order)));

        let mut track_data = Vec::new();
        let mut previous_tick = 0_u64;
        for event in events.into_iter() {
            let delta = event.tick.saturating_sub(previous_tick);
            write_vlq(delta, &mut track_data);
            track_data.extend_from_slice(&event.raw_bytes);
            previous_tick = event.tick;
        }

        out.extend_from_slice(b"MTrk");
        out.extend_from_slice(&(track_data.len() as u32).to_be_bytes());
        out.extend_from_slice(&track_data);
    }

    fs::write(&output_path, out)
        .map_err(|err| format!("Failed writing {}: {err}", output_path.display()))?;

    Ok(output_path)
}

fn next_touchup_path(source_path: &Path) -> PathBuf {
    let parent = source_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = source_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    let ext = source_path
        .extension()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "mid".to_string());

    let mut candidate = parent.join(format!("{stem}_touchup.{ext}"));
    if !candidate.exists() {
        return candidate;
    }

    let mut index = 2_u32;
    loop {
        candidate = parent.join(format!("{stem}_touchup_{index}.{ext}"));
        if !candidate.exists() {
            return candidate;
        }
        index += 1;
    }
}

fn write_vlq(value: u64, out: &mut Vec<u8>) {
    let mut buffer = [0_u8; 10];
    let mut idx = buffer.len();
    let mut v = value;

    idx -= 1;
    buffer[idx] = (v & 0x7F) as u8;
    v >>= 7;

    while v > 0 {
        idx -= 1;
        buffer[idx] = ((v & 0x7F) as u8) | 0x80;
        v >>= 7;
    }

    out.extend_from_slice(&buffer[idx..]);
}

fn encode_midi_message(channel: u8, message: &MidiMessage) -> Vec<u8> {
    let status_channel = channel & 0x0F;
    match message {
        MidiMessage::NoteOff { key, vel } => {
            vec![0x80 | status_channel, key.as_int(), vel.as_int()]
        }
        MidiMessage::NoteOn { key, vel } => vec![0x90 | status_channel, key.as_int(), vel.as_int()],
        MidiMessage::Aftertouch { key, vel } => {
            vec![0xA0 | status_channel, key.as_int(), vel.as_int()]
        }
        MidiMessage::Controller { controller, value } => {
            vec![0xB0 | status_channel, controller.as_int(), value.as_int()]
        }
        MidiMessage::ProgramChange { program } => vec![0xC0 | status_channel, program.as_int()],
        MidiMessage::ChannelAftertouch { vel } => vec![0xD0 | status_channel, vel.as_int()],
        MidiMessage::PitchBend { bend } => {
            let raw = (bend.as_int() as i32 + 0x2000).clamp(0, 0x3FFF) as u16;
            let lsb = (raw & 0x7F) as u8;
            let msb = ((raw >> 7) & 0x7F) as u8;
            vec![0xE0 | status_channel, lsb, msb]
        }
    }
}

fn encode_meta_message(meta: &MetaMessage) -> Vec<u8> {
    let mut out = vec![0xFF];
    match meta {
        MetaMessage::TrackNumber(value) => {
            out.push(0x00);
            if let Some(track_no) = value {
                out.push(0x02);
                out.extend_from_slice(&track_no.to_be_bytes());
            } else {
                out.push(0x00);
            }
        }
        MetaMessage::Text(bytes) => write_meta_payload(&mut out, 0x01, bytes),
        MetaMessage::Copyright(bytes) => write_meta_payload(&mut out, 0x02, bytes),
        MetaMessage::TrackName(bytes) => write_meta_payload(&mut out, 0x03, bytes),
        MetaMessage::InstrumentName(bytes) => write_meta_payload(&mut out, 0x04, bytes),
        MetaMessage::Lyric(bytes) => write_meta_payload(&mut out, 0x05, bytes),
        MetaMessage::Marker(bytes) => write_meta_payload(&mut out, 0x06, bytes),
        MetaMessage::CuePoint(bytes) => write_meta_payload(&mut out, 0x07, bytes),
        MetaMessage::ProgramName(bytes) => write_meta_payload(&mut out, 0x08, bytes),
        MetaMessage::DeviceName(bytes) => write_meta_payload(&mut out, 0x09, bytes),
        MetaMessage::MidiChannel(value) => {
            out.push(0x20);
            out.push(0x01);
            out.push(value.as_int());
        }
        MetaMessage::MidiPort(value) => {
            out.push(0x21);
            out.push(0x01);
            out.push(value.as_int());
        }
        MetaMessage::EndOfTrack => {
            out.push(0x2F);
            out.push(0x00);
        }
        MetaMessage::Tempo(value) => {
            out.push(0x51);
            out.push(0x03);
            let tempo = value.as_int();
            out.push(((tempo >> 16) & 0xFF) as u8);
            out.push(((tempo >> 8) & 0xFF) as u8);
            out.push((tempo & 0xFF) as u8);
        }
        MetaMessage::SmpteOffset(smpte) => {
            out.push(0x54);
            out.push(0x05);
            let fps_bits = match smpte.fps() {
                Fps::Fps24 => 0b00,
                Fps::Fps25 => 0b01,
                Fps::Fps29 => 0b10,
                Fps::Fps30 => 0b11,
            };
            let hour_byte = ((fps_bits << 5) | (smpte.hour() & 0x1F)) as u8;
            out.push(hour_byte);
            out.push(smpte.minute());
            out.push(smpte.second());
            out.push(smpte.frame());
            out.push(smpte.subframe());
        }
        MetaMessage::TimeSignature(numerator, denominator, clocks, notes) => {
            out.push(0x58);
            out.push(0x04);
            out.push(*numerator);
            out.push(*denominator);
            out.push(*clocks);
            out.push(*notes);
        }
        MetaMessage::KeySignature(key, is_minor) => {
            out.push(0x59);
            out.push(0x02);
            out.push(*key as u8);
            out.push(if *is_minor { 1 } else { 0 });
        }
        MetaMessage::SequencerSpecific(bytes) => write_meta_payload(&mut out, 0x7F, bytes),
        MetaMessage::Unknown(kind, bytes) => write_meta_payload(&mut out, *kind, bytes),
    }
    out
}

fn write_meta_payload(out: &mut Vec<u8>, kind: u8, payload: &[u8]) {
    out.push(kind);
    write_vlq(payload.len() as u64, out);
    out.extend_from_slice(payload);
}

fn encode_sysex(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0xF0];
    write_vlq(bytes.len() as u64, &mut out);
    out.extend_from_slice(bytes);
    out
}

fn encode_escape(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0xF7];
    write_vlq(bytes.len() as u64, &mut out);
    out.extend_from_slice(bytes);
    out
}

fn run(cli: Cli) -> (EditorResult, i32) {
    let source = cli.midi.clone();
    let result_state = Arc::new(Mutex::new(EditorResult::cancelled(&source)));

    let app = match MidiTouchupApp::new(
        &source,
        cli.theme.clone(),
        cli.sf2.clone(),
        result_state.clone(),
    ) {
        Ok(app) => app,
        Err(err) => return (EditorResult::error(&source, err), 1),
    };

    let native_options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    let run_result = eframe::run_native(
        "MIDI Touch-Up Editor",
        native_options,
        Box::new(move |_cc| Ok(Box::new(app))),
    );

    if let Err(err) = run_result {
        let report = EditorResult::error(&source, format!("Editor runtime failed: {err}"));
        return (report, 1);
    }

    let final_result = result_state
        .lock()
        .map(|r| r.clone())
        .unwrap_or_else(|_| EditorResult::error(&source, "Failed to read editor result"));

    let exit_code = if final_result.status == "error" { 1 } else { 0 };
    (final_result, exit_code)
}

fn main() {
    let cli = Cli::parse();
    let wants_json = cli.result_json;
    let (result, exit_code) = run(cli);

    if wants_json {
        let payload = serde_json::to_string(&result).unwrap_or_else(|_| {
            let fallback = EditorResult {
                status: "error".to_string(),
                source_path: result.source_path.clone(),
                saved_path: None,
                message: "Failed to serialize result payload".to_string(),
            };
            serde_json::to_string(&fallback).unwrap_or_else(|_| {
                "{\"status\":\"error\",\"source_path\":\"\",\"saved_path\":null,\"message\":\"serialization failure\"}".to_string()
            })
        });
        println!("{payload}");
    }

    std::process::exit(exit_code);
}

#[cfg(test)]
mod ui_policy_tests {
    use super::*;

    fn unique_test_midi(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "synthesia2midi-{name}-{}-{nanos}.mid",
            std::process::id()
        ))
    }

    fn test_app_with_pitches(pitches: &[u8]) -> MidiTouchupApp {
        let source_path = unique_test_midi("octave-app");
        let notes = pitches
            .iter()
            .enumerate()
            .map(|(index, pitch)| EditableNote {
                note_id: index as u64 + 1,
                track_index: 0,
                channel: 0,
                pitch: *pitch,
                start_tick: index as u64 * 120,
                end_tick: index as u64 * 120 + 100,
                velocity_on: 80,
                velocity_off: 64,
                key_lane_unlocked: false,
            })
            .collect();
        let document = MidiDocument {
            source_path: source_path.clone(),
            format_u16: 0,
            division_u16: 480,
            ticks_per_beat: 480,
            tempo_us_per_beat: DEFAULT_TEMPO_US_PER_BEAT,
            tempo_map: TempoMap::default(480),
            notes,
            channel_colors: ChannelColorMap::new(),
            preserved_tracks: vec![Vec::new()],
            max_tick: 3840,
            next_note_id: pitches.len() as u64 + 1,
            dirty: false,
        };
        MidiTouchupApp::from_document(
            document,
            None,
            Arc::new(Mutex::new(EditorResult::cancelled(&source_path))),
        )
    }

    #[test]
    fn wheel_input_only_targets_the_canvas() {
        let canvas = Rect::from_min_max(Pos2::new(100.0, 200.0), Pos2::new(900.0, 700.0));

        assert!(pointer_targets_canvas(
            Some(Pos2::new(400.0, 400.0)),
            canvas
        ));
        assert!(!pointer_targets_canvas(
            Some(Pos2::new(400.0, 100.0)),
            canvas
        ));
        assert!(!pointer_targets_canvas(None, canvas));
    }

    #[test]
    fn toolbar_metrics_contract_for_laptop_widths() {
        let compact = toolbar_metrics(900.0);
        let wide = toolbar_metrics(1500.0);

        assert!(compact.button_width < wide.button_width);
        assert!(compact.button_height < wide.button_height);
        assert!(compact.font_size <= 18.0);
        assert!(compact.button_width <= 120.0);
    }

    #[test]
    fn color_metadata_survives_load_and_touchup_save() {
        use midly::num::{u15, u28};
        use midly::{Format, Header, Timing, TrackEvent};

        let source_path = unique_test_midi("color-map");
        let payload = br#"Synthesia2MIDI:color-map:v1:{"channels":{"0":{"natural":[10,20,30],"sharp_flat":[4,8,12]}}}"#;
        let smf = Smf {
            header: Header::new(Format::SingleTrack, Timing::Metrical(u15::new(480))),
            tracks: vec![vec![
                TrackEvent {
                    delta: u28::new(0),
                    kind: TrackEventKind::Meta(MetaMessage::Text(payload)),
                },
                TrackEvent {
                    delta: u28::new(0),
                    kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
                },
            ]],
        };
        smf.save(&source_path).unwrap();

        let document = load_midi_document(&source_path).unwrap();
        assert_eq!(
            document.channel_colors.get(&0).unwrap().natural,
            Some([10, 20, 30])
        );

        let saved_path = save_midi_document(&document).unwrap();
        let reloaded = load_midi_document(&saved_path).unwrap();
        assert_eq!(
            reloaded.channel_colors.get(&0).unwrap().sharp_flat,
            Some([4, 8, 12])
        );

        let _ = fs::remove_file(source_path);
        let _ = fs::remove_file(saved_path);
    }

    #[test]
    fn octave_plan_moves_every_note_by_twelve() {
        let notes = vec![(1_u64, 48_u8), (2, 72)];

        assert_eq!(
            plan_octave_shift(notes.into_iter(), 1).unwrap(),
            vec![
                PitchChange {
                    note_id: 1,
                    before: 48,
                    after: 60,
                },
                PitchChange {
                    note_id: 2,
                    before: 72,
                    after: 84,
                },
            ]
        );
    }

    #[test]
    fn octave_plan_rejects_entire_shift_at_piano_bounds() {
        assert_eq!(
            plan_octave_shift([(1_u64, 21_u8)].into_iter(), -1),
            Err(OctaveShiftBlock::BelowPiano { pitch: 21 })
        );
        assert_eq!(
            plan_octave_shift([(1_u64, 108_u8)].into_iter(), 1),
            Err(OctaveShiftBlock::AbovePiano { pitch: 108 })
        );
    }

    #[test]
    fn pitch_changes_apply_to_large_documents_without_id_order_assumptions() {
        let pitches = vec![48_u8; 4096];
        let mut app = test_app_with_pitches(&pitches);
        let mut changes = plan_octave_shift(
            app.document
                .notes
                .iter()
                .map(|note| (note.note_id, note.pitch)),
            1,
        )
        .unwrap();
        changes.reverse();

        assert_eq!(
            apply_pitch_changes(&mut app.document.notes, &changes, true),
            pitches.len()
        );
        assert!(app.document.notes.iter().all(|note| note.pitch == 60));

        assert_eq!(
            apply_pitch_changes(&mut app.document.notes, &changes, false),
            pitches.len()
        );
        assert!(app.document.notes.iter().all(|note| note.pitch == 48));
    }

    #[test]
    fn octave_shift_is_one_undoable_redoable_command() {
        let mut app = test_app_with_pitches(&[48, 72]);

        app.apply_octave_shift(1).unwrap();
        assert_eq!(
            app.document
                .notes
                .iter()
                .map(|note| note.pitch)
                .collect::<Vec<_>>(),
            vec![60, 84]
        );
        assert_eq!(app.octave_offset, 1);
        assert_eq!(app.undo_stack.len(), 1);

        app.undo();
        assert_eq!(
            app.document
                .notes
                .iter()
                .map(|note| note.pitch)
                .collect::<Vec<_>>(),
            vec![48, 72]
        );
        assert_eq!(app.octave_offset, 0);

        app.redo();
        assert_eq!(
            app.document
                .notes
                .iter()
                .map(|note| note.pitch)
                .collect::<Vec<_>>(),
            vec![60, 84]
        );
        assert_eq!(app.octave_offset, 1);
    }

    #[test]
    fn blocked_octave_shift_leaves_document_and_history_unchanged() {
        let mut app = test_app_with_pitches(&[21, 60]);

        let result = app.apply_octave_shift(-1);

        assert_eq!(result, Err(OctaveShiftBlock::BelowPiano { pitch: 21 }));
        assert_eq!(
            app.document
                .notes
                .iter()
                .map(|note| note.pitch)
                .collect::<Vec<_>>(),
            vec![21, 60]
        );
        assert!(!app.document.dirty);
        assert_eq!(app.octave_offset, 0);
        assert!(app.undo_stack.is_empty());
        assert!(app.redo_stack.is_empty());
    }

    #[test]
    fn octave_labels_and_boundary_messages_are_user_readable() {
        assert_eq!(octave_offset_label(0), "0");
        assert_eq!(octave_offset_label(1), "+1");
        assert_eq!(octave_offset_label(-2), "-2");
        assert_eq!(midi_pitch_label(21), "A0");
        assert_eq!(midi_pitch_label(60), "C4");
        assert_eq!(midi_pitch_label(108), "C8");

        let low = octave_shift_block_message(OctaveShiftBlock::BelowPiano { pitch: 21 });
        let high = octave_shift_block_message(OctaveShiftBlock::AbovePiano { pitch: 108 });
        assert!(low.contains("A0 (MIDI 21)"));
        assert!(low.contains("A0 and C8"));
        assert!(high.contains("C8 (MIDI 108)"));
        assert!(high.contains("A0 and C8"));
    }
}
