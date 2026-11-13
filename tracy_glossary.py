"""Glossary behind Info > Glossary and the "?" button on every tab.

One entry per graph type, per tab and per recurring term. Each graph entry
says what the graph shows, how the numbers are calculated and how to read it,
in plain words. Keep entries short, define jargon where it is used, and do not
use em dashes.

Entry ids are stable keys the GUI links to:
    viz:<Plot Type>      Visualization plot types (the Plot Type dropdown text)
    behav:<name>         Behavioral Data views
    coh:<name>           Coherence graphs
    spike:<name>         Spike Analysis graphs
    spatial:<key>        Spatial Transients plots (SPATIAL_PLOTS keys)
    bout:<menu label>    Bout Analysis Plot menu entries
    dec:<Output>         Decision Probability outputs (DEC_MODES text)
    kin:<Analysis>       Kinematics analyses (KIN_ANALYSES text)
    link:<name>          Signal Linkage
    tab:<name>           One explainer per tab
    term:<name>          General terms

This module has no Tk dependency so it can be tested headless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GlossaryEntry:
    id: str
    term: str
    where: str
    shows: str
    calc: str = ""
    read: str = ""
    aliases: tuple = field(default_factory=tuple)
    see: tuple = field(default_factory=tuple)

    @property
    def kind(self) -> str:
        prefix = self.id.split(":", 1)[0]
        return {"tab": "tab", "term": "term"}.get(prefix, "graph")


def _clean(text: str) -> str:
    """Collapse the source's line wrapping but keep blank-line paragraphs."""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text.strip())]
    return "\n\n".join(p for p in paras if p)


def _e(id, term, where, shows, calc="", read="", aliases=(), see=()):
    return GlossaryEntry(id, term, where, _clean(shows), _clean(calc),
                         _clean(read), tuple(aliases), tuple(see))


# Section order in the glossary list.
SECTIONS = (
    ("tab", "Tabs"),
    ("viz", "Visualization"),
    ("behav", "Behavioral Data"),
    ("coh", "Coherence"),
    ("spike", "Spike Analysis"),
    ("spatial", "Spike Analysis: Spatial Transients"),
    ("bout", "Bout Analysis"),
    ("dec", "Decision Probability"),
    ("kin", "Kinematics"),
    ("link", "Signal Linkage"),
    ("term", "Terms"),
)

_VIZ = "Data > Visualization > Plot Type"
_BEH = "Data > Behavioral Data > Visualize Data"
_COH_WS = "Data > Coherence > Whole Session"
_COH_BB = "Data > Coherence > By Bout"
_COH_BG = "Data > Coherence > By Group"
_SPK = "Data > Spike Analysis"
_SPA = "Data > Spike Analysis > Spatial Transients..."
_BOUT = "Data > Bout Analysis > Plot"
_DEC = "Data > Decision Probability > Output"
_KIN = "Data > Kinematics > Analysis"
_LINK = "Data > Signal Linkage"

_KIN_COMMON = (
    "Speed comes from the tracked position: distance moved between frames "
    "divided by the time between them, smoothed with a Savitzky-Golay filter "
    "(the Smoothing window and poly order settings). It is a speed, so it "
    "has no direction. The signal is the z-scored channel on the same "
    "timeline.")
_KIN_TBIN = (
    "With a Temporal bin above 0, frames are first averaged into bins of "
    "that many seconds, so each bin (not each frame) is one data point. "
    "This stops thousands of near-identical neighboring frames from "
    "making a weak relationship look significant.")


ENTRIES: tuple = (
    # ── Tabs ─────────────────────────────────────────────────────────────
    _e("tab:Welcome", "Welcome", "Main tab",
       "The landing page: what TRACY is and where to start."),
    _e("tab:Setup", "Project Setup", "Project > Setup",
       """Create or open a project. A project is a folder that holds the
       processed signals, bouts, settings, factors and exclusions for one
       experiment, so everything you do on the other tabs is saved in one
       place.""",
       see=("tab:Processing",)),
    _e("tab:Processing", "Processing", "Project > Processing",
       """Turns raw photometry files into analyzed signals, one subject at a
       time or a whole folder at once.""",
       calc="""For each recording TRACY separates the LED states (415, 470,
       570 nm), fits and removes photobleaching to get dF/F, removes the
       part of the signal that the 415 nm control channel explains (motion
       correction), z-scores the result, lines it up with the behavior or
       position file by clock time, and cuts out a window around every
       scored bout.""",
       see=("term:dF/F", "term:Motion correction", "term:Z-score",
            "term:Bout", "term:Isosbestic (415 nm)")),
    _e("tab:Bout Frames Editor", "Bout Frames Editor",
       "Project > Bout Frames Editor",
       """Import, view and edit the scored behavior bouts (boutframes) and
       TTL events, check how they land on the signal, shift them per subject
       and exclude individual bouts.""",
       calc="""Boutframes are scored in video frames. TRACY converts each
       frame number to a time using the video frame rate and the offset
       between the camera and the photometry rig, then to the nearest
       photometry sample. The Scaled Preview shows where each bout lands
       after that conversion.""",
       see=("term:Bout", "term:Boutframes")),
    _e("tab:Factors", "Factors", "Project > Factors",
       """Describe your experimental design. A factor is one way subjects
       differ (Group, Drug, Sex, Day); each subject gets one level of each
       factor. Every graphing tab can then split, combine or filter on any
       factor.""",
       see=("term:Split series by",)),
    _e("tab:Exclusions", "Exclusions", "Project > Exclusions",
       """Leave bad data out of analyses without deleting it. You can
       exclude a photometry channel for a subject, or a subject's behavior
       data. Each analysis tab has an "Apply exclusions" checkbox that
       decides whether the exclusions are used.""",
       see=("viz:Signal Integrity",)),
    _e("tab:Behavioral Data", "Behavioral Data", "Data > Behavioral Data",
       """A table of behavior measurements per subject from the tracked
       position: time in each zone, entries into each zone, distance
       traveled, speed and decisions. No photometry is involved.""",
       calc="""Each frame is assigned to the zone the animal is in. Very
       short zone visits are merged into the visit before them and short
       tracking gaps are bridged, so a point flickering across a boundary
       is not counted as many entries. Time in a zone is the number of
       frames there divided by the frame rate. An entry is counted when the
       animal stays at least "Min time in zone" and at least "Min time
       between entries" has passed since the last one. Percentages are of
       tracked time. With "Use time bins" the same numbers are computed for
       each bin of the session.""",
       read="""Press Calculate Metrics to fill the table, then Visualize
       Data to plot any column or one of the summary views.""",
       see=("behav:Time in Each Zone", "behav:Entries into Each Zone",
            "behav:Time at Distance from Center", "behav:Single metric")),
    _e("tab:Visualization", "Visualization", "Data > Visualization",
       """The main plotting tab. Pick subjects or groups, choose a Plot
       Type, and press Generate Plot. Plot types run from the raw signal,
       through each processing step, to bout-aligned and position-based
       views.""",
       read="""The settings below Plot Type change with the plot you pick,
       so you only see the options that plot uses. Each plot type has its
       own glossary entry.""",
       see=("viz:Signal Integrity", "viz:Raw Data", "viz:Normalized (dF/F)",
            "viz:Motion Corrected", "viz:Z-scored", "viz:Bouts Overlay",
            "viz:Extracted Bouts", "viz:Bout Length Bins",
            "viz:Compare Across Bouts", "viz:Position Heatmap",
            "viz:Zone Entry Bouts", "viz:Zone Averages",
            "viz:Distance from Center",
            "viz:Distance from Center (Euclidean)", "viz:Out/Back")),
    _e("tab:Coherence", "Coherence", "Data > Coherence",
       """Asks whether two channels (for example two brain regions) rise
       and fall together, and at which rhythms. The left column picks the
       subjects, the two channels, the method and the frequency bands; the
       three pages on the right are three ways to ask the question: over
       the whole session, around bouts, and between groups.""",
       see=("term:Coherence", "coh:Whole Session", "coh:By Bout",
            "coh:By Group", "term:Frequency band",
            "term:Welch vs Morlet wavelet")),
    _e("tab:Spike Analysis", "Spike Analysis", "Data > Spike Analysis",
       """Finds transients (brief upward events, often called spikes) in
       each channel and counts them: how many per minute, how big and how
       wide. Then compares those numbers between groups, zones, bout phases
       and places.""",
       calc="""Two detectors are offered. MAD threshold marks peaks that
       rise a set number of MADs above the session median. Mirror-FDR sets
       each recording's threshold from its own noise, using downward
       excursions as the estimate of how many upward ones are noise. Both
       then drop events that are too narrow, too wide or too close
       together.""",
       read="""Always check the Representative Trace first: if the marked
       events do not look like real events, the summary graphs are not
       meaningful.""",
       see=("spike:Representative Trace", "spike:Visualize Spike Data",
            "spike:Analyze Spikes by Zone", "spike:Spikes by Bout Phase",
            "spatial:breakdown", "term:MAD threshold", "term:Mirror-FDR",
            "term:Transient")),
    _e("tab:Bout Analysis", "Bout Analysis", "Data > Bout Analysis",
       """Turns each bout's signal into numbers (average, peak, area,
       decay) and compares them between subjects, groups, channels, bout
       order and bout length. Also runs time-course statistics that test
       the whole trace rather than one number.""",
       calc="""Calculate Metrics reads the z-scored window around every
       bout onset and measures it inside the analysis window (set in
       Settings, in seconds from onset). All Plot menu graphs are drawn
       from that table.""",
       see=("term:Bout metrics", "bout:Bar Graphs (per metric)",
            "bout:Compare Across Bouts",
            "bout:Compare Channels (within groups)",
            "bout:Compare First vs Last", "bout:Binned Progression",
            "bout:Bars by Bout Length",
            "bout:Time-Course Statistics (FLMM-style)")),
    _e("tab:Decision Probability", "Decision Probability",
       "Data > Decision Probability",
       """Asks whether the signal says anything about what the animal does
       next: go on into the risky zone (explore) or go back to the safe one
       (avoid or retreat). You choose the safe zone (From) and the risky
       zone (To); the Output dropdown picks one of four ways to ask.""",
       read="""The first output scores every frame; the other three score
       one row per actual decision and test it against a shifted-signal
       null, which is the stricter test.""",
       see=("dec:Signal-binned probability",
            "dec:Explore vs avoid (center)", "dec:Signal vs P(explore)",
            "dec:Distal turn-back", "term:Circular-shift null",
            "term:AUROC")),
    _e("tab:Kinematics", "Kinematics (BETA)", "Data > Kinematics",
       """Relates how the animal moves (speed, acceleration, position) to
       the signal. The Analysis dropdown offers twenty views of that
       relationship; each has its own glossary entry.""",
       calc=_KIN_COMMON + "\n\n" + _KIN_TBIN,
       see=("kin:Velocity vs Signal (overlay)",
            "kin:Velocity–Signal cross-correlation",
            "kin:Signal by velocity bin", "kin:Arena heatmap: Signal",
            "kin:Movement-onset triggered signal",
            "term:Velocity and acceleration")),
    _e("tab:Signal Linkage", "Signal Linkage (BETA)", _LINK,
       """Asks, for each behavior, how tightly the start of a bout is tied
       to a fast change in the signal, and whether that happens on most
       bouts or only on average.""",
       see=("link:Onset linkage",)),
    _e("tab:Overview", "Info: Overview", "Info > Overview",
       """Reference pages: required files and naming, the recommended
       workflow and troubleshooting."""),
    _e("tab:Glossary", "Info: Glossary", "Info > Glossary",
       """This page. Search any graph, tab or term. The "?" button on each
       tab opens the entry for the graph you have selected there."""),
    _e("tab:Changelog", "Info: Changelog", "Info > Changelog",
       "What changed in each version."),
    _e("tab:System Check", "Info: System Check", "Info > System Check",
       "Checks that the Python packages TRACY needs are installed."),

    # ── Visualization ────────────────────────────────────────────────────
    _e("viz:Signal Integrity", "Signal Integrity", _VIZ,
       """A quality score from 0 to 100 for each channel: is there real
       biological signal here, or noise, dropouts and movement artifact?
       One subject gives a dashboard of the pieces behind the score;
       several subjects give a color-coded table with one row per channel.""",
       calc="""Photobleaching is flattened first. In the default (GCaMP)
       mode the 415 nm control is fitted to the 470 nm signal and removed.
       Four sub-scores, each 0 to 100, are then measured:

       Dynamics: how large the slow fluctuations (slower than about 1.5 s,
       where real events live) are compared with the fast frame-to-frame
       jitter. This carries the most weight.

       Noise: how much of the variance is frame-to-frame jitter.

       Dropout: penalizes signals that mostly spike downward or have
       extreme outliers, the mark of a loose fiber or connector.

       Motion: penalizes a signal that closely follows the 415 nm channel
       or loses most of its variance when 415 is removed, and sudden level
       shifts.

       The score is a weighted mix (40% dynamics, 20% noise, 18% dropout,
       22% motion). Any one severe failure also caps the whole score, so
       three good sub-scores cannot hide a fatal one. Sensor mode ignores
       415 and scores dynamics, noise and dropout only. Finally, if the
       channel responds at scored bouts more strongly than at the same
       bouts shifted to random times, the score is raised to at least 70
       (up to 90); this evidence can only raise a score, never lower it.""",
       read="""Roughly: 70 and above is usable, under 50 is suspect. A star
       in the table means the score was raised by a clear response at
       bouts. Use "Auto-exclude below score" to send low channels to the
       Exclusions tab.""",
       aliases=("SI", "quality score", "signal quality"),
       see=("term:Isosbestic (415 nm)", "tab:Exclusions",
            "term:Circular-shift null")),
    _e("viz:Raw Data", "Raw Data", _VIZ,
       """The fluorescence exactly as recorded, split by LED: one trace for
       415 nm, one for 470 nm and one for 570 nm on each ticked channel.""",
       calc="""The rig alternates LEDs frame by frame. TRACY sorts the
       frames by the LedState column into one trace per wavelength. Nothing
       else is done.""",
       read="""Expect a slow downward drift (photobleaching). Sudden steps
       or flat lines point to a hardware problem that later steps cannot
       repair.""",
       see=("viz:Normalized (dF/F)", "term:Isosbestic (415 nm)")),
    _e("viz:Normalized (dF/F)", "Normalized (dF/F)", _VIZ,
       """The signal as a percent change from its own baseline, with the
       slow photobleaching drift removed.""",
       calc="""A smooth decaying curve (two exponentials, optionally three)
       is fitted to each raw trace to capture bleaching. dF/F is then
       100 x (raw - fitted) / fitted. If the curve cannot be fitted TRACY
       falls back to a straight-line fit.""",
       read="""The trace should now sit around zero with no overall slope.
       Units are percent.""",
       see=("term:dF/F", "viz:Raw Data", "viz:Motion Corrected")),
    _e("viz:Motion Corrected", "Motion Corrected", _VIZ,
       """dF/F after removing whatever the 415 nm control channel also
       shows, which is mostly movement and blood-flow artifact.""",
       calc="""The 415 nm dF/F trace is scaled to best match the signal
       trace with a robust straight-line fit (one that ignores outliers),
       and that scaled copy is subtracted. What remains is the part of the
       signal 415 cannot explain.""",
       read="""Compare with Normalized (dF/F): shared wobbles should be
       gone and real events should remain.""",
       see=("term:Motion correction", "term:Isosbestic (415 nm)",
            "viz:Z-scored")),
    _e("viz:Z-scored", "Z-scored", _VIZ,
       """The motion-corrected signal rescaled so that 0 is the session
       average and 1 is one standard deviation. This is the trace every
       later analysis uses.""",
       calc="""z = (value - mean) / standard deviation, using the mean and
       standard deviation of that channel over the session (or over the
       pooled sessions when pooled z-scoring is turned on).""",
       read="""Because every channel is on the same scale, z-scores can be
       averaged across animals whose raw brightness differs.""",
       see=("term:Z-score", "viz:Bouts Overlay")),
    _e("viz:Bouts Overlay", "Bouts Overlay", _VIZ,
       """The whole z-scored session with every scored bout marked on it,
       so you can see what the signal did when the behavior happened.""",
       calc="""Each bout's start (and end, when scored) is converted from
       video frames to a time on the signal and drawn as a shaded span or
       ribbon. "Select Behaviors" overlays several behaviors at once.""",
       read="""Look for the signal changing at the shaded spans more than
       elsewhere. This is the best view for checking that bouts are aligned
       before trusting any average.""",
       see=("term:Bout", "viz:Extracted Bouts")),
    _e("viz:Extracted Bouts", "Extracted Bouts", _VIZ,
       """Every bout of one behavior cut out and lined up at its onset
       (time 0): the individual traces, their mean with a shaded SEM band,
       and a heatmap with one row per bout. One row of panels per ticked
       channel.""",
       calc="""For each bout the z-scored signal from "pre" seconds before
       onset to "post" seconds after is copied out. The mean is the average
       of those windows at each time point; SEM is their standard deviation
       divided by the square root of the count. With "Average bouts within
       subject" each animal is averaged first so it counts once.""",
       read="""A consistent rise or dip after time 0 in both the mean and
       most heatmap rows is a real response. A mean driven by one or two
       bright rows is not.""",
       aliases=("peri-event", "PSTH", "bout traces"),
       see=("term:Mean and SEM", "viz:Compare Across Bouts",
            "viz:Bout Length Bins")),
    _e("viz:Bout Length Bins", "Bout Length Bins", _VIZ,
       """Bout-aligned mean traces, drawn separately for short, medium and
       long bouts.""",
       calc="""Each bout's length is its scored end minus its start. Bouts
       are sorted into the length bins you define (default under 1 s, 1 to
       3 s, over 3 s; edit them from Bout Analysis > Plot > Edit Length
       Bins) and the windows in each bin are averaged.""",
       read="""If long bouts show a larger or longer response than short
       ones, the signal follows the behavior's duration. Needs boutframes
       with end frames.""",
       see=("bout:Bars by Bout Length", "viz:Extracted Bouts")),
    _e("viz:Compare Across Bouts", "Compare Across Bouts", _VIZ,
       """Bout-aligned traces separated by bout number, to show whether
       the response changes from the first bout to the last.""",
       calc="""Bouts are numbered in the order they happened. "Bouts per
       bin" groups them (3 gives bouts 1-3, 4-6, ...; a list such as 1-2,
       4-5 gives custom bins). Within each bin the windows are averaged
       across animals and drawn as mean with SEM. The heatmap can be
       ordered by subject or by bout number.""",
       read="""Lines that shrink or grow steadily with bout number show
       habituation or sensitisation within the session.""",
       see=("bout:Binned Progression", "bout:Compare First vs Last")),
    _e("viz:Position Heatmap", "Position Heatmap", _VIZ,
       """A map of the arena colored by the average signal recorded while
       the animal was at each spot.""",
       calc="""The arena is divided into squares of "Bin Size (cm)". For
       each square TRACY averages the z-score of every frame the animal
       spent there. Squares never visited are left blank. The zone outlines
       can be drawn on top.""",
       read="""Warm squares are places where the signal ran high. A square
       visited for only a moment rests on very few frames, so read isolated
       bright squares with care.""",
       see=("viz:Zone Averages", "kin:Arena heatmap: Signal")),
    _e("viz:Zone Entry Bouts", "Zone Entry Bouts", _VIZ,
       """Signal lined up to the moment the animal enters a zone (for
       example an open arm), treated like a bout onset.""",
       calc="""An entry is detected from the position track when the animal
       crosses into the chosen zone type, stays at least "Min time in
       zone", and at least "Min time between entries" has passed since the
       previous entry. The z-scored window around each entry is cut out and
       averaged as in Extracted Bouts.""",
       read="""Time 0 is the crossing. Activity before 0 is the approach;
       after 0 is time in the zone.""",
       see=("viz:Extracted Bouts", "viz:Zone Averages")),
    _e("viz:Zone Averages", "Zone Averages", _VIZ,
       """One bar per zone type: the average signal while the animal was in
       that zone.""",
       calc="""Every frame is labeled with the zone the animal is in. The
       z-scores of all frames in each zone category are averaged per
       channel. With several subjects each animal is one dot and the bar is
       the mean with SEM across animals.""",
       read="""A taller bar means the signal was higher, on average,
       throughout the time spent in that zone. It does not say when during
       a visit the signal rose.""",
       see=("viz:Position Heatmap", "viz:Zone Entry Bouts")),
    _e("viz:Distance from Center", "Distance from Center", _VIZ,
       """Average signal against how far the animal is from the arena
       center, drawn separately for the X direction and the Y direction.""",
       calc="""For every frame the distance from center is taken along X
       and along Y. Frames are grouped into distance bins ("Distance bin
       (cm)", default 1 cm) and the z-scores in each bin are averaged.""",
       read="""On an elevated plus maze the arms lie along X and Y, so the
       two panels are the two arm types. A line rising with distance means
       the signal grows as the animal goes further out.""",
       see=("viz:Distance from Center (Euclidean)", "viz:Out/Back")),
    _e("viz:Distance from Center (Euclidean)",
       "Distance from Center (Euclidean)", _VIZ,
       """Average signal against straight-line distance from the arena
       center, in any direction.""",
       calc="""Distance is the straight line from the center to the
       animal, sqrt(x^2 + y^2). Frames are grouped into distance bins and
       their z-scores averaged, as in Distance from Center.""",
       read="""Use this for an open field, where direction does not matter
       and only how peripheral the animal is does.""",
       see=("viz:Distance from Center",
            "kin:Distance-from-center vs Signal")),
    _e("viz:Out/Back", "Out/Back", _VIZ,
       """Average signal while the animal moves outward along an open arm
       compared with while it moves back toward the center.""",
       calc="""Only open-arm frames are used. A frame counts as Out when
       the animal's speed away from the center exceeds "Min velocity", and
       as Back when its speed toward the center exceeds it. Slower frames
       are ignored. The z-scores of the Out frames and of the Back frames
       are each averaged.""",
       read="""A higher Out bar means the signal is higher on the way out
       than on the retreat. "Show Velocity Histogram" shows which frames
       the threshold keeps.""",
       see=("viz:Out/Back velocity histogram",
            "dec:Distal turn-back")),
    _e("viz:Out/Back velocity histogram", "Out/Back velocity histogram",
       "Data > Visualization > Out/Back > Show Velocity Histogram",
       """How fast animals move along the open arms, and how much data a
       given Out/Back threshold would keep.""",
       calc="""Left: a histogram of every open-arm frame's speed along the
       arm, signed so that positive is away from the center, pooled over
       all processed subjects, with the ignored band shaded. Right: the
       share of open-arm frames that pass each possible threshold.""",
       read="""Pick a threshold that removes the pile of near-zero speeds
       (the animal sitting still) without throwing away most of the
       movement.""",
       see=("viz:Out/Back",)),

    # ── Behavioral Data ──────────────────────────────────────────────────
    _e("behav:Time in Each Zone", "[Summary] Time in Each Zone", _BEH,
       "How long each subject or group spent in each zone type.",
       calc="""Frames in each zone are counted and divided by the frame
       rate to give seconds. "Zone View: simple" merges sub-zones (for
       example proximal and distal open arm) into open, center and closed;
       "complex" keeps them apart. In Group mode bars are the mean with SEM
       across animals.""",
       see=("tab:Behavioral Data", "behav:Entries into Each Zone")),
    _e("behav:Entries into Each Zone", "[Summary] Entries into Each Zone",
       _BEH,
       "How many times each subject or group entered each zone type.",
       calc="""An entry is a crossing into the zone that lasts at least
       "Min time in zone" and comes at least "Min time between entries"
       after the last one. Counts are per zone category.""",
       see=("tab:Behavioral Data", "behav:Time in Each Zone")),
    _e("behav:Time at Distance from Center",
       "[Summary] Time at Distance from Center", _BEH,
       """How many seconds the animal spent at each distance from the arena
       center (straight-line, along X, or along Y).""",
       calc="""Each tracked frame's distance from center is put into a bin
       of "Distance bin (cm)". Frames per bin divided by the frame rate
       gives seconds. Thin lines are subjects; the thick line is the mean
       with SEM. A subject that never reached a far bin counts as 0 s
       there.""",
       read="""This is behavior only. Compare it with Visualization >
       Distance from Center, which shows the signal at those distances.""",
       see=("viz:Distance from Center",)),
    _e("behav:Single metric", "Single metric plot", _BEH,
       """Any one column of the metrics table drawn as a histogram or bar
       per subject, a bar per group, or a line over time bins.""",
       calc="""The plotted values are the table's values. Histogram shows
       how subjects are spread; bar shows each subject, or the group mean
       with SEM; line (with "Use time bins") shows the metric in each bin
       of the session.""",
       see=("tab:Behavioral Data",)),

    # ── Coherence ────────────────────────────────────────────────────────
    _e("coh:Whole Session", "Whole Session coherence", _COH_WS,
       """Coherence between the two chosen channels over the entire
       recording. Run Analysis computes up to three things (static
       coherence, sliding coherence, rolling correlation); the buttons
       beside it draw them.""",
       calc="""The two channels' z-scored traces for the full session are
       compared with the chosen Method. In Subject mode each subject is
       analyzed on its own; in Group mode each subject is analyzed and the
       results are averaged within each series. The "Long Session" and
       "Short Assay" presets set window lengths suited to recordings of
       tens of minutes or of a few minutes.""",
       see=("coh:Visualize Coherence", "coh:Power Spectra",
            "coh:Wavelet Spectrogram", "coh:Band Bars (whole session)",
            "term:Coherence")),
    _e("coh:Visualize Coherence", "Visualize Coherence", _COH_WS,
       """The main whole-session figure: a static coherence spectrum,
       coherence over time, and rolling correlation over time.""",
       calc="""Static coherence: the session is cut into overlapping
       windows; in each the two signals are broken into frequencies, and
       coherence at a frequency is how consistently the two keep the same
       relationship there across windows (0 = unrelated, 1 = locked).

       Sliding coherence: the same measure inside a window that slides
       along the session, so it becomes a value over time.

       Rolling correlation: the ordinary Pearson correlation of the two
       z-scored traces inside a sliding window. Unlike coherence it is
       signed, so -1 means the channels move opposite to each other.""",
       read="""A peak in the spectrum says the regions share a rhythm at
       that frequency. The table's Mean Coherence and Mean Correlation
       summarize the same curves.""",
       see=("term:Coherence", "term:Welch vs Morlet wavelet",
            "coh:Wavelet Spectrogram")),
    _e("coh:Power Spectra", "Power Spectra", _COH_WS,
       """How much each channel fluctuates at each frequency, one curve per
       channel.""",
       calc="""Each channel's z-scored trace is cut into overlapping
       windows, each window is broken into frequencies, and the strength at
       each frequency is averaged over windows (Welch's method, with the
       same settings as static coherence).""",
       read="""Coherence only means something at frequencies where both
       channels have power. Check here before trusting a coherence peak.""",
       see=("coh:Visualize Coherence",)),
    _e("coh:Wavelet Spectrogram", "Wavelet Spectrogram", _COH_WS,
       """A heatmap of coherence with time across and frequency up: when
       during the session the two channels were coupled, and at which
       rhythm.""",
       calc="""This is the sliding-coherence result drawn as an image. Each
       column is a moment in the session and each row a frequency; color is
       coherence from 0 to 1.""",
       read="""Horizontal bands are rhythms coupled throughout; isolated
       patches are brief episodes of coupling.""",
       see=("coh:Visualize Coherence", "coh:Epoch Spectrogram")),
    _e("coh:Band Bars (whole session)", "Band Bars (whole session)",
       _COH_WS,
       "Whole-session coherence summarized as one bar per frequency band.",
       calc="""The static coherence spectrum is averaged over the
       frequencies inside each band from the Frequency bands list. One bar
       per subject or series; group bars are the mean with SEM.""",
       see=("term:Frequency band", "coh:Visualize Coherence")),
    _e("coh:By Bout", "By Bout (epoch) coherence", _COH_BB,
       """Coherence just before a behavior starts compared with just after
       it starts. Zone entries can be used in place of scored bouts.""",
       calc="""For every bout onset two windows are taken: "Seconds before
       onset" (baseline) and "Seconds after onset". Coherence is computed
       in each window and averaged over a subject's bouts. Bouts shorter
       than "Min bout length" can be dropped so the post window is not
       mostly some other behavior.""",
       see=("coh:Plot Epoch Coherence", "coh:Epoch Band Bars",
            "coh:Epoch Spectrogram")),
    _e("coh:Plot Epoch Coherence", "Plot Epoch Coherence", _COH_BB,
       """Two coherence spectra on one plot: baseline (before onset) and
       post-onset.""",
       calc="""Each subject's bouts are averaged into one baseline spectrum
       and one post-onset spectrum; the lines are the mean with SEM across
       the selected subjects.""",
       read="""Where the post line sits above the baseline line, the
       behavior brought the two regions into step at that frequency.""",
       see=("coh:By Bout", "coh:Epoch Band Bars")),
    _e("coh:Epoch Band Bars", "Plot Band Bars (epochs)", _COH_BB,
       "Baseline and post-onset coherence as a pair of bars per frequency band.",
       calc="""The baseline and post-onset spectra are each averaged within
       every frequency band.""",
       see=("term:Frequency band", "coh:Plot Epoch Coherence")),
    _e("coh:Epoch Spectrogram", "Epoch Spectrogram", _COH_BB,
       """A time-by-frequency heatmap of coherence around bout onset, with
       onset at time 0.""",
       calc="""The signal from the pre window through the post window is
       cut out around every bout. A wavelet transform gives coherence at
       each moment and frequency, and these maps are averaged over
       bouts.""",
       read="""A patch that appears right after time 0 is coupling that
       switches on with the behavior.""",
       see=("coh:Wavelet Spectrogram", "coh:Peri-Event Spectrogram")),
    _e("coh:By Group", "By Group coherence", _COH_BG,
       """Compares coherence between experimental groups, either over the
       whole session or as a before/after change around bouts.""",
       calc="""Coherence is computed for each subject separately, exactly
       as on the Whole Session or By Bout page ("Comparison type" picks
       which). Subjects are then divided into series by the "Split series
       by" column, and each series is summarized as the mean with SEM
       across its subjects. The optional test (t-test or Mann-Whitney U)
       compares series using one value per subject.""",
       see=("coh:Plot Spectra (groups)", "coh:Plot Band Bars (groups)",
            "coh:Zone Coherence", "coh:Plot Spectra (Pre vs Post)",
            "coh:Plot Band Bars (Pre vs Post)",
            "coh:Plot Delta Coherence", "coh:Peri-Event Spectrogram")),
    _e("coh:Plot Spectra (groups)", "Plot Spectra (groups)", _COH_BG,
       "Whole-session coherence spectra, one mean line per group.",
       calc="""Static coherence is computed per subject; each group's line
       is the mean across its subjects with an SEM band. Individual
       subjects can be overlaid.""",
       see=("coh:Visualize Coherence",)),
    _e("coh:Plot Band Bars (groups)", "Plot Band Bars (groups)", _COH_BG,
       """Group mean coherence in each frequency band, with an optional
       statistical test between groups.""",
       calc="""Each subject's spectrum is averaged within each band. Bars
       are the group mean with SEM. The Stats test dropdown runs a t-test
       or Mann-Whitney U between groups per band.""",
       see=("term:Frequency band",)),
    _e("coh:Zone Coherence", "Zone Coherence", _COH_BG,
       """Coherence between the two channels while the animal is in each
       zone type, compared across groups.""",
       calc="""For each subject the frames spent in each zone category are
       joined together and static coherence is computed on them. Bars show
       the mean coherence per zone, grouped.""",
       read="""Because separate visits are joined end to end, zones with
       very little time give unreliable values.""",
       see=("viz:Zone Averages",)),
    _e("coh:Plot Spectra (Pre vs Post)", "Plot Spectra (Pre vs Post)",
       _COH_BG,
       """For each group, the baseline and post-onset coherence spectra
       side by side.""",
       calc="""Bout-epoch coherence is computed for every subject, then
       averaged by group. Post-onset is the solid line; the baseline it is
       compared against is dashed.""",
       see=("coh:Plot Epoch Coherence",)),
    _e("coh:Plot Band Bars (Pre vs Post)", "Plot Band Bars (Pre vs Post)",
       _COH_BG,
       """Post-onset coherence per frequency band, one panel per band,
       groups side by side with a dot per subject.""",
       calc="""Each subject's post-onset spectrum (averaged over its bouts)
       is averaged within each frequency band. Bars are the group mean with
       SEM, and the chosen test compares groups within each band.""",
       see=("coh:Plot Delta Coherence", "term:Frequency band")),
    _e("coh:Plot Delta Coherence", "Plot Delta Coherence", _COH_BG,
       """The change in coherence caused by the behavior (post-onset minus
       baseline) per frequency band and group.""",
       calc="""For each subject and band: post-onset band coherence minus
       baseline band coherence. Bars are group means with SEM, with the
       chosen test between groups.""",
       read="""Above zero means the behavior increased coupling; below
       zero means it decreased it.""",
       see=("coh:Plot Band Bars (Pre vs Post)",)),
    _e("coh:Peri-Event Spectrogram", "Peri-Event Spectrogram", _COH_BG,
       """One time-by-frequency coherence map per group around bout onset,
       plus an optional difference map between two groups.""",
       calc="""As Epoch Spectrogram, but averaged over all subjects of a
       group. The difference map is group A minus group B, drawn in a
       two-color scale centered on zero.""",
       see=("coh:Epoch Spectrogram",)),

    # ── Spike Analysis ───────────────────────────────────────────────────
    _e("spike:Representative Trace", "Representative Trace",
       _SPK + " > Plot Trace",
       """One subject's signal with the baseline, the detection threshold
       and every detected transient marked.""",
       calc="""Drawn with the detection settings as currently typed. With
       Mirror-FDR the trace is the smoothed signal the detector scores,
       with the raw signal faint behind it, and the title reports the
       estimated share of detections that are noise.""",
       read="""The markers should sit on events you would circle by eye.
       Many markers on flat noise means the threshold is too low; obvious
       events without markers means it is too high.""",
       see=("term:MAD threshold", "term:Mirror-FDR", "term:Transient")),
    _e("spike:Visualize Spike Data", "Visualize Spike Data",
       _SPK + " > Visualize Spike Data",
       """Bar graphs of transient rate and mean amplitude for each channel,
       by subject or group. A single subject is shown as its trace with the
       transients marked.""",
       calc="""Rate is the number of detected transients divided by the
       recording length in minutes. Amplitude is the mean height of the
       detected peaks (above the local baseline with Mirror-FDR). Group
       bars are the mean with SEM across animals.""",
       see=("tab:Spike Analysis", "term:Transient")),
    _e("spike:Analyze Spikes by Zone", "Analyze Spikes by Zone",
       _SPK + " > Analyze Spikes by Zone",
       """Transient rate, amplitude and width in each zone type, for every
       channel.""",
       calc="""Each detected transient is assigned to the zone the animal
       was in at its peak. Rate in a zone is the transients there divided
       by the minutes spent there. Amplitude and width are averaged over
       the transients in that zone. Sub-zones pool as total transients over
       total time.""",
       read="""Rate already accounts for time spent, so a zone the animal
       rarely visits can still show a high rate; with little time there it
       rests on few events.""",
       see=("spatial:breakdown", "viz:Zone Averages")),
    _e("spike:Spikes by Bout Phase", "Spikes by Bout Phase",
       _SPK + " > Spikes by Bout Phase",
       """Transient rate before, during and after a behavior's bouts, per
       group.""",
       calc="""For each bout three stretches are defined: Pre (the window
       before onset), During (onset to the scored end, or the zone visit
       for a zone entry) and Post (the window after the end). For every
       bout the transients in each stretch are counted and divided by that
       stretch's length to give transients per minute, and these are
       averaged over the animal's bouts. Each animal gives one rate per
       phase; bars are the group mean with SEM.""",
       read="""A During bar above Pre means events become more frequent
       while the behavior is happening.""",
       see=("term:Bout",)),

    # ── Spatial Transients ───────────────────────────────────────────────
    _e("spatial:breakdown", "Rate breakdown (zone / still-moving / time)",
       _SPA,
       """One row of bars per channel that slices the same measure many
       ways: whole session, each zone type, still versus moving, each time
       block of the session, and a same-time-budget rate. Bars are the mean
       with SEM; each dot is one animal.""",
       calc="""The chosen metric (transients per minute, percent of time in
       a transient, or median amplitude) is computed over the frames in
       each slice. "Still" is slower than the still-speed setting. The
       same-time-budget rate recomputes each animal's rate as if every
       animal had spent the same share of time in each condition, which
       removes differences that come only from where animals chose to be.""",
       read="""If one group is lower in every slice, the difference is not
       explained by where the animals went or how much they moved.""",
       see=("term:Spatial transient metrics",
            "spike:Analyze Spikes by Zone")),
    _e("spatial:distance", "Distance from center (transients)", _SPA,
       """The chosen transient metric against distance from the arena
       center (straight-line, |X| or |Y|).""",
       calc="""Frames are grouped into distance bins and the metric is
       computed in each (bins with under 3 s of data are left blank). Zone
       types can be included, excluded or drawn as separate lines. "Correct
       for speed and time" fits a model in which the rate depends on place,
       speed and time into the session, and reports only the place part, so
       a place is not marked quiet just because animals run through it.""",
       read="""With the correction on and "relative" ticked, 1.5 means 50%
       more transients at that distance than speed and time alone
       predict.""",
       see=("viz:Distance from Center", "term:Spatial transient metrics")),
    _e("spatial:speed", "Speed tuning", _SPA,
       "The chosen transient metric at each running speed.",
       calc="""Frames are grouped by the animal's smoothed speed into fixed
       bands (0-1, 1-2, 2-4, ... cm/s) and the metric is computed in each.
       Lines are the mean with SEM across animals.""",
       read="""Transient rate usually falls as speed rises. That is why the
       place plots offer a speed correction.""",
       see=("kin:Signal by velocity bin",)),
    _e("spatial:scatter", "Transient scatter (x/y)", _SPA,
       """A map with one dot at the animal's position for every transient,
       or a rate map.""",
       calc="""Dots are drawn faintly so that dense areas build up their
       own heatmap. "Divide by time spent" switches to transients per
       second of time in each square, which removes the part of the picture
       that is only where the animal sat.""",
       read="""Without dividing by time, the dots mostly show where the
       animal spent its time.""",
       see=("viz:Position Heatmap",)),
    _e("spatial:trace3d", "3D trace across the maze", _SPA,
       """The animal's path drawn on the maze floor, lifted up wherever the
       signal was high.""",
       calc="""Height is a heavily smoothed z-score, with each transient
       boosted by a gain times its amplitude so events stand out. The path
       is colored by height.""",
       read="""A picture for showing where peaks happened in one or two
       recordings. Height is in arbitrary units and is not for
       measurement."""),

    # ── Bout Analysis ────────────────────────────────────────────────────
    _e("bout:Bar Graphs (per metric)", "Bar Graphs (per metric)", _BOUT,
       """One bar chart per ticked metric: the mean with SEM for each
       subject or group.""",
       calc="""Each bout gives one value per metric. In Subject mode a bar
       is the mean over that subject's bouts. In Group mode each animal's
       bouts are averaged first, then animals are averaged, so the SEM
       reflects animals and not bouts.""",
       see=("term:Bout metrics", "term:Mean and SEM")),
    _e("bout:Compare Across Bouts", "Compare Across Bouts (metrics)", _BOUT,
       """Each metric drawn bout by bout: bout 1, bout 2, and so on, with a
       separate line per group.""",
       calc="""The metric is taken for each bout in the order bouts
       happened. At each bout number the values are averaged across
       subjects (mean with SEM).""",
       read="""A slope across bout numbers shows the response changing
       with repetition. Later bout numbers hold fewer animals, so their
       error bars widen.""",
       see=("viz:Compare Across Bouts", "bout:Binned Progression")),
    _e("bout:Compare Channels (within groups)",
       "Compare Channels (within groups)", _BOUT,
       """Each channel's mean bout-aligned trace on the same axes, one
       panel per group.""",
       calc="""For every channel the bout windows are averaged within each
       animal, then across animals (mean with SEM), and drawn as one
       colored line per channel.""",
       read="""Shows whether two regions respond to the same behavior with
       the same timing and direction.""",
       see=("viz:Extracted Bouts",)),
    _e("bout:Compare First vs Last", "Compare First vs Last", _BOUT,
       """The average of each subject's first X bouts beside the average of
       its last X bouts, for every metric.""",
       calc="""X is "Bouts per bin (X)". For each subject the metric is
       averaged over the first X bouts and over the last X bouts. Bars are
       the mean with SEM across subjects and the paired values are tested
       against each other.""",
       read="""A drop from First to Last is habituation; a rise is
       sensitisation.""",
       see=("bout:Binned Progression",)),
    _e("bout:Binned Progression", "Binned Progression", _BOUT,
       """A line graph of the response across the session in bins of X
       bouts (bouts 1-X, X+1-2X, ...).""",
       calc="""Each bout's peak, mean and area under the curve are measured
       in the analysis window. Within a subject they are averaged over the
       bouts in each bin; points are the mean with SEM across subjects.""",
       read="""The same question as Compare Across Bouts, with less noise
       because several bouts are averaged per point.""",
       see=("bout:Compare Across Bouts", "bout:Compare First vs Last")),
    _e("bout:Bars by Bout Length", "Bars by Bout Length", _BOUT,
       """Each metric for short, medium and long bouts, as bars per length
       bin.""",
       calc="""Each bout's length is its scored end minus its start. Bouts
       are sorted into the length bins (Plot > Edit Length Bins) and the
       metric is averaged within each bin.""",
       read="""Needs boutframes with end frames. A response that grows
       with bout length follows how long the behavior lasted.""",
       see=("viz:Bout Length Bins",)),
    _e("bout:Time-Course Statistics (FLMM-style)",
       "Time-Course Statistics (FLMM-style)", _BOUT,
       """A statistical test at every time point of the bout window,
       rather than on one summary number. The plot is an effect curve over
       time with two bands around it, and marks the times where the effect
       is reliably different from zero.""",
       calc="""At each time point a mixed model is fitted to all bouts,
       with a random intercept per animal so that an animal's many bouts
       are not treated as independent. The curve, written beta(t), is the
       estimated effect at each time. You choose what the effect is:

       Signal not 0: is the mean z-score different from baseline?

       Across bouts: does the signal change with bout number?

       Between groups: do two groups differ?

       Across-bouts x group: does the change with bout number differ
       between groups?

       Behavior contrast, Factor, Pairwise matrix: does one behavior
       differ from another, from a reference, or every behavior from every
       other (corrected for the number of pairs)?

       The inner band is the ordinary 95% interval at each time point. The
       outer, simultaneous band is widened to account for testing many
       time points at once.""",
       read="""Trust stretches where the outer band excludes zero. Where
       only the inner band excludes zero the effect is suggestive but could
       be a chance finding from looking at many time points. Finished runs
       are kept in this menu for the session.""",
       aliases=("FLMM", "FUI", "fastFMM", "functional mixed model",
                "beta(t)"),
       see=("bout:FLMM peak effect", "term:Mixed model",
            "viz:Extracted Bouts")),
    _e("bout:FLMM peak effect", "FLMM per-subject peak effect",
       "Data > Bout Analysis > Time-Course Statistics > Peak effect",
       """A violin per behavior with one dot per animal: each animal's
       signal at the moment the group effect peaks.""",
       calc="""The peak time and the significant window come from the
       group-level time-course test for that behavior against the control.
       Each animal's own mean trace is then read at that one time point.
       The time is taken from the group, not from each animal's own
       maximum, so the values are not biased upward.""",
       read="""Dots clearly away from the control's dots show the effect
       is present in most animals, not just in the average.""",
       see=("bout:Time-Course Statistics (FLMM-style)",)),

    # ── Decision Probability ─────────────────────────────────────────────
    _e("dec:Signal-binned probability", "Signal-binned probability", _DEC,
       """For each level of the signal, the chance that the animal moves
       on to a given zone within the next few seconds. One curve for
       Forward (From to To, explore) and one for Reverse.""",
       calc="""Every frame in which the animal is in the starting zone is
       scored: what is the z-score now, and does the animal enter the
       outcome zone within the look-ahead window (default 10 s)? Frames
       are grouped by z-score into bins (default 0.5 wide from -3 to 3),
       and in each bin the probability is the share of frames followed by
       an entry. With "Subject-mean" each animal gets its own curve and the
       plot is their mean with SEM; "Pool all frames" mixes all animals'
       frames first. Bins with too few frames are hidden.""",
       read="""A rising curve means the animal is more likely to make that
       move when the signal is high. Because every frame is scored and
       neighboring frames are nearly identical, this curve looks more
       certain than it is; the event outputs are the stricter test.""",
       see=("dec:Explore vs avoid (center)", "tab:Decision Probability")),
    _e("dec:Explore vs avoid (center)", "Explore vs avoid (center)", _DEC,
       """Each time the animal comes from the safe zone into the center it
       makes one decision: go on to the risky zone (explore) or return
       (avoid). This output asks whether the signal differs between those
       two kinds of visit. Three panels.""",
       calc="""A decision is a center visit entered from the From zone and
       left to either From (avoid) or To (explore). Zone visits shorter
       than "Ignore zone visits under" are merged away first. For each
       decision the z-score is averaged over the chosen window ("Center
       signal from": the second before entry, the whole center visit, or
       its last second).

       Left panel: each animal's mean signal on avoid visits joined to its
       mean on explore visits.

       Middle panel: each animal's AUROC, the chance that a randomly picked
       explore visit has a higher signal than a randomly picked avoid
       visit. 0.5 is no difference.

       Right panel: the mean signal around center entry for explore
       (solid) and avoid (dashed).

       The group mean AUROC is tested against the same decisions scored on
       each animal's own signal shifted in time by a random amount (at
       least 30 s), repeated "Shift-null repeats" times.""",
       read="""Lines that mostly slope the same way and AUROC dots that
       sit away from 0.5 mean the signal separates the two choices. Trust
       p(shift) over the t-test. Only the "Before center entry" window is
       a prediction; the other two windows overlap the choice itself.""",
       see=("term:AUROC", "term:Circular-shift null",
            "dec:Signal vs P(explore)")),
    _e("dec:Signal vs P(explore)", "Signal vs P(explore)", _DEC,
       """Whether the animal explores more often when the signal is
       higher: probability of exploring at low, middle and high signal.""",
       calc="""The same center decisions as Explore vs avoid. Within each
       animal the decisions are sorted by their signal and split into three
       equal parts. For each third, the x value is the mean z-score and the
       y value is the share of those decisions that were explore. Thin
       lines are animals (at least 9 decisions each); the thick line is the
       group mean with SEM.""",
       read="""A line that climbs from left to right means a higher signal
       goes with more exploring. A flat line means no relationship.""",
       see=("dec:Explore vs avoid (center)",)),
    _e("dec:Distal turn-back", "Distal turn-back", _DEC,
       """Looks at trips into the risky zone and the point where the
       animal turned around. Asks whether the signal just before turning
       back is special. Three panels.""",
       calc="""An excursion is one visit to the To zone that goes at least
       "Min excursion depth" further out than where it began. Its turn-back
       point is the frame furthest from the center. The signal is averaged
       over the window just before that frame.

       Left panel: one dot per excursion, signal before turning back
       against how far out the animal got. Filled dots went back to the
       safe zone next; open dots returned to the risky zone.

       Middle panel: a same-place control. For each excursion TRACY finds
       the same animal's longer excursions and takes the signal as it
       passed that same distance while still going out. Each animal's line
       joins "kept going" to "turned back".

       Right panel: the mean signal around the turn-back point.

       Both the turned-back minus kept-going difference and the
       within-animal correlation between signal and distance are tested against
       the shifted-signal null.""",
       read="""If "turned back" sits above "kept going" at the same
       distance, the signal is tied to the decision to retreat and not just
       to being far out.""",
       see=("term:Circular-shift null", "viz:Out/Back")),

    # ── Kinematics ───────────────────────────────────────────────────────
    _e("kin:Velocity vs Signal (overlay)", "Velocity vs Signal (overlay)",
       _KIN,
       """Speed and the signal drawn over the session on one plot, speed on
       the left axis and z-score on the right.""",
       calc=_KIN_COMMON + """ The title reports Pearson r between the two
       traces.""",
       read="""Look for the two traces moving together or in opposition. A
       strongly negative r means the signal drops when the animal runs.""",
       see=("kin:Velocity vs Signal (scatter)", "term:Pearson r")),
    _e("kin:Velocity vs Signal (scatter)", "Velocity vs Signal (scatter)",
       _KIN,
       """Each point is one moment: speed across, signal up, with a fitted
       straight line.""",
       calc=_KIN_TBIN + """ The line is an ordinary least-squares fit.
       Pearson r measures a straight-line relationship; Spearman rho
       measures any steadily rising or falling one.""",
       read="""A cloud sloping down means faster movement goes with lower
       signal.""",
       see=("kin:Signal by velocity bin", "term:Pearson r")),
    _e("kin:Velocity–Signal cross-correlation",
       "Velocity–Signal cross-correlation", _KIN,
       """How strongly speed and signal are related when one is shifted in
       time against the other, to find which leads.""",
       calc="""The correlation between speed and signal is computed with
       the signal shifted by every lag from -Max lag to +Max lag. Lag 0 is
       the same moment; a positive lag means the signal is taken after the
       movement. Each animal's delay is the lag of its own strongest
       correlation; a one-sample t-test asks whether the delays differ from
       0, and groups are compared on delay and depth. This runs at full
       frame resolution.""",
       read="""A dip at a positive lag means the signal change follows the
       movement change by that many seconds. The red dot marks the most
       negative group-mean value; ticks at the bottom are individual
       animals.""",
       see=("kin:Velocity–Signal coupling (summary bars)",)),
    _e("kin:Velocity–Signal coupling (summary bars)",
       "Velocity–Signal coupling (summary bars)", _KIN,
       """Three single-number summaries of how strongly speed and signal
       are tied, as bars per group with a dot per animal.""",
       calc="""From each animal's cross-correlation: the value at lag 0,
       the mean value over lags 0 to 2 s, and the area of the negative part
       at positive lags. Stars test a group against 0; brackets test groups
       against each other.""",
       read="""Negative values mean inverse coupling. Use this when the
       cross-correlation is too flat for its dip's timing to be
       trusted.""",
       see=("kin:Velocity–Signal cross-correlation",)),
    _e("kin:Signal by velocity bin", "Signal by velocity bin", _KIN,
       "The average signal at each speed.",
       calc="""Speed is divided into "# bins" equal ranges (or ranges of
       speed z-score when "Bin by velocity z-score" is ticked) and the
       z-score is averaged in each. Points are the mean with SEM; in Group
       mode, across animals.""",
       read="""A staircase going down means the signal falls steadily as
       the animal moves faster. Flat means no dependence on speed.""",
       see=("spatial:speed",)),
    _e("kin:Acceleration vs Signal (overlay)",
       "Acceleration vs Signal (overlay)", _KIN,
       "Acceleration and the signal drawn over the session on one plot.",
       calc="""Acceleration is the rate of change of speed: positive when
       speeding up, negative when slowing down. The title reports
       Pearson r.""",
       see=("term:Velocity and acceleration",)),
    _e("kin:Acceleration vs Signal (scatter)",
       "Acceleration vs Signal (scatter)", _KIN,
       "Each point is one moment: acceleration across, signal up.",
       calc=_KIN_TBIN,
       read="""A positive r means the signal is higher while speeding up;
       negative, while slowing down.""",
       see=("term:Velocity and acceleration",)),
    _e("kin:Signal by acceleration bin", "Signal by acceleration bin", _KIN,
       "The average signal at each level of acceleration.",
       calc="""Acceleration is divided into "# bins" ranges, from strongly
       slowing to strongly speeding up, and the z-score is averaged in
       each (mean with SEM).""",
       read="""Shows whether the signal follows changes in speed rather
       than speed itself."""),
    _e("kin:Accelerating vs decelerating", "Accelerating vs decelerating",
       _KIN,
       "Two bars: mean signal while speeding up and while slowing down.",
       calc="""Frames with acceleration above 0 and frames below 0 are
       averaged separately per animal. With more than one animal the two
       are compared with a paired t-test.""",
       read="""A taller decelerating bar suggests the signal rises as the
       animal stops."""),
    _e("kin:Radial velocity vs Signal", "Radial velocity vs Signal", _KIN,
       """Signal against how fast the animal is moving toward or away from
       the arena center.""",
       calc="""Radial velocity is the part of the movement that points
       along the line from the center to the animal: negative when heading
       inward, positive when heading outward. It is plotted against the
       z-score with a correlation.""",
       read="""A relationship here ties the signal to approaching the
       center versus heading for the edge.""",
       see=("viz:Out/Back",)),
    _e("kin:Center vs Perimeter coupling", "Center vs Perimeter coupling",
       _KIN,
       """The speed-signal correlation computed separately for the inner
       and outer parts of the arena.""",
       calc="""Frames are split at the animal's median distance from
       center into an inner half and an outer half. Pearson r between speed
       and signal is computed in each.""",
       read="""Compares whether movement and signal are tied more tightly
       in the center than along the edge."""),
    _e("kin:Arena heatmap: Signal", "Arena heatmap: Signal", _KIN,
       "A map of the arena colored by the mean signal at each spot.",
       calc="""The arena is divided into squares of "Spatial bin (cm)" and
       the z-score of all frames in each square is averaged. Unvisited
       squares are blank. Color limits are set in Graph Settings.""",
       see=("viz:Position Heatmap",)),
    _e("kin:Arena heatmap: Velocity", "Arena heatmap: Velocity", _KIN,
       "A map of the arena colored by the mean speed at each spot.",
       calc="Speed is averaged over the frames in each square.",
       read="Shows where the animal runs and where it lingers."),
    _e("kin:Arena heatmap: Acceleration", "Arena heatmap: Acceleration",
       _KIN,
       "A map of the arena colored by the mean acceleration at each spot.",
       calc="Acceleration is averaged over the frames in each square.",
       read="Shows where the animal tends to speed up and slow down."),
    _e("kin:3D arena map (height=velocity, color=signal)",
       "3D arena map (height=velocity, color=signal)", _KIN,
       """A 3D surface over the arena: height is mean speed at each spot,
       color is mean signal there.""",
       calc="""Speed and z-score are each averaged per square of the
       arena. Unvisited squares are left blank.""",
       read="""Tall areas are fast areas; the color on top says what the
       signal was doing there."""),
    _e("kin:Velocity×Acceleration phase map (signal)",
       "Velocity×Acceleration phase map (signal)", _KIN,
       """A grid with speed across and acceleration up, colored by the mean
       signal for each combination.""",
       calc="""Every frame is placed by its speed and acceleration into a
       "# bins" by "# bins" grid and the z-scores in each cell are
       averaged.""",
       read="""The bottom-left corner (slow and slowing) is where the
       animal is coming to a stop; the right side is fast movement."""),
    _e("kin:Movement-onset triggered signal",
       "Movement-onset triggered signal", _KIN,
       "The average signal around the moments the animal starts moving.",
       calc="""A movement onset is the frame where speed rises above
       "Movement threshold (cm/s)" after being below it. The signal around
       every onset is lined up at time 0 and averaged (mean with SEM over
       onsets). This always uses full frame resolution.""",
       read="""A dip after 0 means the signal falls when movement begins;
       a change before 0 means the signal moves first.""",
       see=("viz:Extracted Bouts",)),
    _e("kin:Distance-from-center vs Signal",
       "Distance-from-center vs Signal", _KIN,
       "The average signal at each distance from the arena center.",
       calc="""Frames are grouped by straight-line distance from center
       and the z-score is averaged in each group (mean with SEM).""",
       see=("viz:Distance from Center (Euclidean)",)),
    _e("kin:Kinematic–Signal coupling by zone",
       "Kinematic–Signal coupling by zone", _KIN,
       """One bar per zone: how strongly speed (or acceleration) and the
       signal are correlated while the animal is in that zone.""",
       calc="""Frames are labeled by zone. Within each zone the
       correlation between the chosen kinematic ("Zone-plot kinematic") and
       the z-score is computed.""",
       read="Compare bars to see where movement and signal are most tied."),
    _e("kin:Velocity & acceleration distribution",
       "Velocity & acceleration distribution", _KIN,
       """Histograms of speed and of acceleration with a smooth outline, a
       tick for every data point, and dotted lines at the mean and 1, 2
       and 3 standard deviations.""",
       calc="""Each point is one temporal bin. With groups selected each
       group gets its own row of panels.""",
       read="""Shows how the animal moves overall and what speed a "+2 SD"
       event corresponds to."""),

    # ── Signal Linkage ───────────────────────────────────────────────────
    _e("link:Onset linkage", "Onset linkage (table and traces)", _LINK,
       """For each behavior: a plain-language verdict, a table of numbers
       and a panel with the mean bout-aligned trace. The verdict is Tightly
       linked, Population-linked (variable) or Not linked.""",
       calc="""All bouts of a behavior are lined up at onset.

       dF/dt: the steepest slope of the smoothed mean trace within "Search"
       seconds of onset, in z per second. It measures how fast the signal
       changes, not how big it gets.

       Delta: the mean signal in the "Response" window after onset minus
       the mean in the "Baseline" window before it.

       Latency: when that steepest slope occurs relative to onset.
       Negative means the signal turned before the behavior started.

       Reliability r: the average correlation between every pair of
       individual smoothed bout traces around onset; how alike single
       bouts are.

       Locking Z and p: the steepest slope compared with the slopes found
       when the same number of windows are taken at random times in the
       recording (the "Permutations" setting).

       Linkage Index (0 to 100) combines the locking test and the
       reliability in equal parts.""",
       read="""Tightly linked needs both a significant locking test and
       consistent single bouts. Population-linked means the average
       responds but individual bouts vary. In the trace panels the dashed
       line is onset, the dotted line is the steepest slope and the shaded
       area is the response window.""",
       aliases=("Linkage Index", "locking", "dF/dt", "reliability"),
       see=("viz:Extracted Bouts", "term:Circular-shift null")),

    # ── Terms ────────────────────────────────────────────────────────────
    _e("term:Z-score", "Z-score", "Used on every Data tab",
       """A signal rescaled so 0 is its average and 1 is one standard
       deviation. A z-score of 2 means "two standard deviations above this
       channel's usual level".""",
       calc="z = (value - mean) / standard deviation.",
       aliases=("z", "zscore", "standardized"),
       see=("viz:Z-scored",)),
    _e("term:dF/F", "dF/F", "Processing, Visualization",
       """Change in fluorescence relative to baseline, in percent. It
       removes differences in overall brightness between fibers and
       animals.""",
       calc="100 x (F - baseline) / baseline, where baseline is the fitted bleaching curve.",
       aliases=("delta F over F", "deltaF/F", "normalized"),
       see=("viz:Normalized (dF/F)",)),
    _e("term:Isosbestic (415 nm)", "Isosbestic (415 nm)",
       "Processing, Signal Integrity",
       """Light at 415 nm excites GCaMP the same amount whether or not
       calcium is bound, so the 415 nm trace shows movement and blood-flow
       artifact without neural activity. It is the control the signal is
       corrected against. For GRAB-type sensors 415 nm is not a clean
       control; use Sensor mode.""",
       aliases=("415", "control channel", "reference channel"),
       see=("term:Motion correction", "viz:Signal Integrity")),
    _e("term:Motion correction", "Motion correction", "Processing",
       """Removing from the signal whatever the 415 nm control channel also
       shows.""",
       calc="""The 415 nm trace is scaled to best fit the signal and
       subtracted.""",
       see=("viz:Motion Corrected", "term:Isosbestic (415 nm)")),
    _e("term:Bout", "Bout", "Bout Frames Editor and every bout plot",
       """One occurrence of a scored behavior, with a start and (if scored)
       an end. Bout-aligned plots line up all bouts at their start, called
       onset or time 0. The pre and post bout windows set how many seconds
       before and after onset are kept.""",
       aliases=("onset", "epoch", "event", "trial"),
       see=("term:Boutframes", "viz:Extracted Bouts")),
    _e("term:Boutframes", "Boutframes", "Project > Bout Frames Editor",
       """The spreadsheet of scored behavior: one sheet per subject, one
       column per behavior, each cell the video frame where a bout starts
       (or ends).""",
       see=("term:Bout", "tab:Bout Frames Editor")),
    _e("term:Mean and SEM", "Mean and SEM", "Most graphs",
       """The mean is the average. SEM (standard error of the mean) is how
       uncertain that average is: the standard deviation divided by the
       square root of the number of values. Error bars and shaded bands in
       TRACY are SEM unless a graph says otherwise.""",
       read="""SEM shrinks as more values are averaged. Check whether the
       values are bouts or animals: averaging bouts within each animal
       first gives the honest, wider band.""",
       aliases=("error bars", "shaded band", "standard error")),
    _e("term:Bout metrics", "Bout metrics", "Data > Bout Analysis",
       """The numbers measured from each bout inside the analysis window
       (seconds from onset, set in Settings).""",
       calc="""Avg: mean z-score in the window.

       Max and Min: highest and lowest z-score in the window.

       First derivative: the steepest change between neighboring samples,
       in z per second.

       AUC: area between the trace and zero, in z x seconds. "+only" counts
       only the part above zero; "-only" only the part below.

       Tau: decay time constant in seconds. An exponential decay is fitted
       from the peak to the end of the window; tau is the time to fall to
       about 37% of the peak. Left blank when the decay cannot be fitted
       reliably.

       t-half: time from the peak until the signal has fallen halfway back
       to baseline, read directly from the smoothed trace with no curve
       fitting.""",
       aliases=("AUC", "tau", "t-half", "t half", "peak", "area under the curve"),
       see=("tab:Bout Analysis", "bout:Bar Graphs (per metric)")),
    _e("term:Coherence", "Coherence", "Data > Coherence",
       """A number from 0 to 1, for each frequency, that says how
       consistently two signals keep the same relationship at that rhythm.
       1 means that whenever one channel oscillates at that frequency the
       other does too, with a fixed timing between them; 0 means no
       consistent relationship.""",
       read="""Coherence ignores sign and delay: two channels that move
       exactly opposite, or one always half a second behind the other, are
       still fully coherent. Rolling correlation is the measure that keeps
       the sign.""",
       see=("coh:Visualize Coherence", "term:Frequency band")),
    _e("term:Frequency band", "Frequency band", "Data > Coherence",
       """A named range of frequencies (for example 0.1 to 0.5 Hz) over
       which coherence is averaged for bar graphs and statistics. Edit them
       in the Frequency bands list. Photometry signals are slow, so the
       bands are far below the EEG bands of the same names.""",
       see=("term:Coherence",)),
    _e("term:Welch vs Morlet wavelet", "Welch vs Morlet wavelet",
       "Data > Coherence > Method",
       """Two ways to break a signal into frequencies. Welch cuts the
       signal into fixed-length windows and averages them: stable, but
       every frequency gets the same time resolution. Morlet wavelet uses
       short windows for fast rhythms and long windows for slow ones, which
       follows changes over time better and is the default.""",
       see=("term:Coherence",)),
    _e("term:Transient", "Transient (spike)", "Data > Spike Analysis",
       """A brief upward event in the signal. Photometry records the summed
       activity of many cells, so a transient is a population event, not a
       single neuron's action potential. Rate is transients per minute;
       amplitude is peak height; width is how long the event lasts at half
       its height.""",
       aliases=("spike", "event", "calcium transient"),
       see=("tab:Spike Analysis",)),
    _e("term:MAD threshold", "MAD threshold", "Data > Spike Analysis",
       """A detection threshold set from the signal's own spread. MAD
       (median absolute deviation) is the typical distance of samples from
       the median, and is not inflated by the transients themselves.""",
       calc="""Threshold = median + k x MAD, where k is "Threshold (MAD)".
       Peaks above it that also stand out from their surroundings by "Min
       prominence" are kept.""",
       see=("term:Mirror-FDR", "spike:Representative Trace")),
    _e("term:Mirror-FDR", "Mirror-FDR", "Data > Spike Analysis",
       """A detection method that sets each recording's threshold from its
       own noise. Real transients only go up; noise goes up and down
       equally. So the number of downward bumps of a given size estimates
       how many upward bumps of that size are just noise.""",
       calc="""Heights are measured above a running-median baseline. The
       threshold is the lowest height at which downward bumps are no more
       than the "FDR target" share of upward ones (0.10 means about 1 in 10
       detections may be noise). If no height achieves that, the channel is
       reported as having nothing that stands out from noise.""",
       aliases=("FDR", "false discovery rate"),
       see=("term:MAD threshold", "spike:Representative Trace")),
    _e("term:Spatial transient metrics", "Spatial transient metrics", _SPA,
       "The three measures the Spatial Transients plots can show.",
       calc="""Transients / min: events divided by time.

       Time in transient (%): the share of time the signal is inside an
       event (around each peak, as wide as the event at half height). Less
       noisy than counts when events are few.

       Median amplitude: the typical event height.""",
       see=("spatial:breakdown",)),
    _e("term:Circular-shift null", "Circular-shift null (p(shift))",
       "Decision Probability, Signal Integrity, Signal Linkage",
       """A way to ask "could this have happened by chance?" that respects
       how slowly photometry signals move.""",
       calc="""The behavior events are kept exactly where they are, and
       the animal's own signal is slid along in time by a random amount,
       wrapping the end around to the start. The statistic is recomputed,
       and this is repeated hundreds of times. p(shift) is the share of
       shifted results at least as extreme as the real one.""",
       read="""Shifting keeps the signal's slow drifts and the behavior's
       clustering, and breaks only their alignment. Ordinary tests assume
       every event is independent and so tend to be too optimistic.""",
       aliases=("shift null", "permutation", "p(shift)", "shuffle")),
    _e("term:AUROC", "AUROC", "Data > Decision Probability",
       """A number from 0 to 1 that says how well a signal separates two
       kinds of event. Here: the chance that a randomly chosen explore
       decision has a higher signal than a randomly chosen avoid
       decision.""",
       read="""0.5 means no separation. 1 means the signal is always
       higher before exploring; 0 means always lower.""",
       aliases=("AUC ROC", "area under the ROC curve"),
       see=("dec:Explore vs avoid (center)",)),
    _e("term:Pearson r", "Pearson r and Spearman rho", "Kinematics, Coherence",
       """Correlation from -1 to 1. Pearson r measures how well a straight
       line describes the relationship. Spearman rho does the same on
       ranks, so it picks up any steadily rising or falling relationship
       and is less affected by outliers.""",
       aliases=("correlation", "rho")),
    _e("term:Velocity and acceleration", "Velocity and acceleration",
       "Data > Kinematics",
       """In TRACY "velocity" is speed: how fast the animal moves, with no
       direction. Acceleration is the rate of change of speed: positive
       when speeding up, negative when slowing down. Neither depends on
       which way the animal is facing.""",
       aliases=("speed",),
       see=("tab:Kinematics",)),
    _e("term:Mixed model", "Mixed model (random intercept)",
       "Bout Analysis statistics",
       """A statistical model for data where each animal contributes many
       measurements. It lets each animal have its own baseline level (the
       random intercept), so twenty bouts from one animal are not counted
       as twenty independent animals.""",
       see=("bout:Time-Course Statistics (FLMM-style)",)),
    _e("term:Split series by", "Split series by (factors)",
       "Group mode on every Data tab",
       """Decides how the selected subjects are divided into the lines or
       bars of a plot. For each factor: "split" draws one series per level,
       "combine" pools the levels, and picking a single level keeps only
       those subjects.""",
       aliases=("facet", "group", "factor"),
       see=("tab:Factors",)),
    _e("term:Apply exclusions", "Apply exclusions", "Every Data tab",
       """When ticked, channels and subjects marked on the Exclusions tab
       are left out of that tab's plots, statistics and exports.""",
       see=("tab:Exclusions",)),
)


_BY_ID = {e.id: e for e in ENTRIES}
if len(_BY_ID) != len(ENTRIES):
    raise ValueError("duplicate glossary id")


def entry(entry_id):
    """The entry with this id, or None."""
    return _BY_ID.get(entry_id)


def section_of(e):
    """Section title an entry is listed under."""
    prefix = e.id.split(":", 1)[0]
    return dict(SECTIONS).get(prefix, "Other")


def ordered_entries():
    """Every entry in display order: by section, then alphabetically."""
    rank = {p: i for i, (p, _t) in enumerate(SECTIONS)}
    return sorted(ENTRIES, key=lambda e: (rank.get(e.id.split(":", 1)[0], 99),
                                          e.term.lower()))


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def search(query):
    """Entries matching every word of ``query``, best match first.

    A match in the term outranks one in an alias, which outranks one in the
    body, so typing a graph's name lands on that graph. An empty query returns
    everything in display order.
    """
    words = _norm(query or "").split()
    if not words:
        return ordered_entries()
    scored = []
    for pos, e in enumerate(ordered_entries()):
        term = _norm(e.term)
        alias = _norm(" ".join(e.aliases))
        body = _norm(" ".join((e.where, e.shows, e.calc, e.read)))
        score = 0
        for w in words:
            if term.startswith(w) or f" {w}" in f" {term}":
                score += 100
            elif w in term:
                score += 60
            elif w in alias:
                score += 40
            elif w in body:
                score += 5
            else:
                score = None
                break
        if score is None:
            continue
        if term == " ".join(words):
            score += 500
        scored.append((-score, pos, e))
    return [e for _s, _p, e in sorted(scored, key=lambda t: t[:2])]


def resolve(prefix, label, fallback=None):
    """Id for a UI selector value, e.g. ``resolve('viz', 'Z-scored')``.

    Returns ``fallback`` when the selector holds something the glossary has no
    entry for, so a "?" button always opens somewhere sensible.
    """
    eid = f"{prefix}:{label}"
    return eid if eid in _BY_ID else fallback
