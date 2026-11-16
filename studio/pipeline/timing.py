from dataclasses import dataclass
from fractions import Fraction
import math

@dataclass(frozen=True)
class TimingDecision:
    source_audio_seconds: float
    video_seconds: float
    tempo: float
    duration_multiplier: float
    needs_review: bool
    reason: str

def fit_audio(audio_seconds: float, video_seconds: float, min_tempo=.85, max_tempo=1.15):
    if not all(math.isfinite(x) and x>0 for x in (audio_seconds,video_seconds)):
        raise ValueError('Audio and video durations must be finite and positive.')
    if not 0<min_tempo<=1<=max_tempo:
        raise ValueError('Tempo limits must include normal speed.')
    tempo=audio_seconds/video_seconds
    review=not min_tempo<=tempo<=max_tempo
    return TimingDecision(audio_seconds,video_seconds,tempo,1/tempo,review,
        'Choose another complete action range or review the audio adjustment.' if review else 'Within the configured automatic audio adjustment range.')

def atempo_filters(tempo):
    if not math.isfinite(tempo) or tempo<=0:raise ValueError('Invalid tempo')
    factors=[]
    while tempo<.5:factors.append(.5);tempo/=.5
    while tempo>2:factors.append(2);tempo/=2
    factors.append(tempo)
    return ','.join(f'atempo={x:.12g}' for x in factors)

def allocate_frames(durations, fps='30000/1001'):
    rate=Fraction(fps);elapsed=0.;last=0;result=[]
    for d in durations:
        if not math.isfinite(d) or d<=0:raise ValueError('Invalid shot duration')
        elapsed+=d;end=round(elapsed*float(rate))
        if end<=last:raise ValueError('Shot shorter than one frame')
        result.append((last,end));last=end
    return result

def pacing_outliers(tempos,max_adjacent_ratio=1.2):
    return [i for i in range(1,len(tempos)) if max(tempos[i],tempos[i-1])/min(tempos[i],tempos[i-1])>max_adjacent_ratio]

