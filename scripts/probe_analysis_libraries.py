"""3-way probe: librosa 1.0 vs Essentia vs the repo's own physical_dsp_v2."""
import json
from pathlib import Path

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so
# the script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"

import numpy as np
import librosa, essentia, essentia.standard as es
import librosa.feature.rhythm as lfrhythm
from copilot.audio.physical_dsp_v2.pipeline import analyze_path, AnalyzerFamily

for a in ('warningActive', 'infoActive'):
    try: setattr(essentia.log, a, False)
    except Exception: pass

KS_MAJOR = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
KS_MINOR = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def ks_key(chroma):
    best = None
    for i in range(12):
        for profile, scale in ((KS_MAJOR, 'major'), (KS_MINOR, 'minor')):
            r = float(np.corrcoef(np.roll(profile, i), chroma)[0, 1])
            if best is None or r > best[0]:
                best = (r, NOTES[i], scale)
    return best[1], best[2], best[0]


def note_of(f0):
    if not np.isfinite(f0) or f0 <= 0: return None
    n = int(round(12 * np.log2(f0 / 440.0))) + 69
    return f'{NOTES[n % 12]}{n // 12 - 1}'


def probe(path):
    name = Path(path).name
    print('=' * 78); print(f'ARCHIVO: {name}')
    out = {'file': name}

    # ---------- librosa ----------
    y, sr = librosa.load(path, sr=None, mono=True)
    t = lfrhythm.tempo(y=y, sr=sr, aggregate=np.median)[0]
    bt_tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    key, scale, r = ks_key(chroma)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    yh, yp = librosa.effects.hpss(y)
    f0, vf, vp = librosa.pyin(y, fmin=float(librosa.note_to_hz('C1')), fmax=float(librosa.note_to_hz('C6')), sr=sr)
    f0v = f0[np.isfinite(f0)]
    sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    labels = librosa.segment.agglomerative(librosa.feature.chroma_cqt(y=y, sr=sr), k=6)
    print('  [librosa 1.0]')
    print(f'    tempo feature.rhythm      {float(t):8.2f} BPM   | beat_track {float(np.atleast_1d(bt_tempo)[0]):8.2f} BPM ({len(beats)} beats)')
    print(f'    tonalidad (KS manual)     {key} {scale}   (r={r:.3f}) | detuning {tuning:+.3f} semitonos')
    print(f'    onsets                    {len(onsets):4d} ({len(onsets)/ (len(y)/sr):.2f}/s)')
    print(f'    HPSS percusivo/harmonico  {float(np.sqrt(np.mean(yp**2)))/float(np.sqrt(np.mean(yh**2))):.3f}')
    print(f'    pyin f0 mediana           {note_of(float(np.median(f0v))) if f0v.size else None} ({float(np.median(f0v)) if f0v.size else 0:.1f} Hz, voiced {float(np.mean(vf)):.2f})')
    print(f'    centroid {sc:8.1f} Hz | flatness {flat:.5f} | RMS {rms:.5f}')
    print(f'    segmentacion (k=6)        {len(set(labels.tolist()))} segmentos')
    out['librosa'] = {'bpm': round(float(t), 2), 'key': f'{key} {scale}', 'tuning': round(float(tuning), 3),
                      'onsets': len(onsets), 'hpss_ratio': round(float(np.sqrt(np.mean(yp**2))/np.sqrt(np.mean(yh**2))), 4),
                      'pyin_median_note': note_of(float(np.median(f0v))) if f0v.size else None,
                      'centroid_hz': round(sc, 1), 'segments': len(set(labels.tolist()))}

    # ---------- essentia ----------
    mono44 = np.mean(np.asarray(es.AudioLoader(filename=path)()[0], dtype=np.float32), axis=1)
    st = es.AudioLoader(filename=path)()
    bpm, ticks, conf, _, _ = es.RhythmExtractor2013(method='multifeature')(mono44)
    ekey, escale, estrength = es.KeyExtractor()(mono44)
    _, _, integ, lra = es.LoudnessEBUR128(sampleRate=st[1])(st[0])
    eonsets, erate = es.OnsetRate()(mono44)
    mel = es.PredominantPitchMelodia()(mono44)[0]
    melv = mel[np.isfinite(mel) & (mel > 0)]
    print('  [essentia 2.1-beta6]')
    print(f'    RhythmExtractor2013       {bpm:8.2f} BPM   (confianza {conf:.3f}, {len(ticks)} ticks)')
    print(f'    KeyExtractor              {ekey} {escale}  (strength {estrength:.3f})')
    print(f'    LoudnessEBUR128           {integ:8.2f} LUFS | LRA {lra:.2f} LU')
    print(f'    OnsetRate                 {len(eonsets):4d} ({erate:.2f}/s)')
    print(f'    PredominantPitchMelodia   {note_of(float(np.median(melv))) if melv.size else None} ({float(np.median(melv)) if melv.size else 0:.1f} Hz)')
    print(f'    Danceability              {es.Danceability()(mono44)[0]:.4f}')
    out['essentia'] = {'bpm': round(float(bpm), 2), 'key': f'{ekey} {escale}', 'lufs': round(float(integ), 2),
                       'lra': round(float(lra), 2), 'onsets': len(eonsets),
                       'melodia_note': note_of(float(np.median(melv))) if melv.size else None}

    # ---------- repo (physical_dsp_v2) ----------
    bundle = analyze_path(path, analyzers=(AnalyzerFamily.TONAL.value, AnalyzerFamily.LEVEL_DYNAMICS.value,
                                           AnalyzerFamily.TRANSIENTS.value, AnalyzerFamily.RHYTHM.value),
                          use_cache=False, cache_dir=Path('logs/pdsp_probe_cache'))
    tonal = bundle.by_analyzer(AnalyzerFamily.TONAL.value)[0]
    level = bundle.by_analyzer(AnalyzerFamily.LEVEL_DYNAMICS.value)[0]
    trans = bundle.by_analyzer(AnalyzerFamily.TRANSIENTS.value)[0]
    print('  [repo physical_dsp_v2 (numpy/scipy/pyloudnorm)]')
    cands = tonal.get('key_candidates').value or []
    print(f'    candidatos de key         {[c["label"] for c in cands[:3]]} | seleccionada=None (por diseño)')
    print(f'    LUFS (pyloudnorm)         {level.get("integrated_loudness").value} | RMS {level.get("rms").value:.5f}')
    tvals = [v for v in trans.values]
    print(f'    transients                {tvals[0].value if tvals else None} | keys={[v.name for v in tvals][:4]} | {trans.limitation_codes()[:2]}')
    print(f'    limitaciones tonal        {tonal.limitation_codes()}')
    out['repo'] = {'key_candidates': [c['label'] for c in cands[:3]],
                   'lufs_pyloudnorm': level.get('integrated_loudness').value,
                   'rms': level.get('rms').value,
                   'tonal_limitations': tonal.limitation_codes()}
    return out


res = [probe(p) for p in ['logs/captures/aasi_v1_MANUAL_224_256_BASS_cfc312d4a47d.wav',
                          'logs/captures/aasi_v1_MANUAL_224_256_Main_with_BASS_cfc312d4a47d.wav']]
Path('logs/librosa_vs_essentia_vs_repo_v1.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
print('\nEVIDENCIA -> logs/librosa_vs_essentia_vs_repo_v1.json')
