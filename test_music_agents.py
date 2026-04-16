from marble.environments.music_env import MusicEvaluationEnvironment
from marble.agent.music_agent import *

music_features = {

    "pitch_var": 0.8,
    "harmony_complexity": 0.7,
    "tempo": 120,
    "rhythm_stability": 0.8,
    "section_count": 5,
    "structure_variation": 0.7,
    "valence": 0.6,
    "energy": 0.7,
    "novelty": 0.8,
    "genre_fusion": 0.6,
    "clarity": 0.7,
    "feature_consistency": 0.8
}

env = MusicEvaluationEnvironment(music_features)
music_features = {

    "pitch_var": 0.8,
    "harmony_complexity": 0.7,

    "tempo": 120,
    "rhythm_stability": 0.8,

    "section_count": 5,
    "structure_variation": 0.7,

    "valence": 0.6,
    "energy": 0.7,

    "novelty": 0.8,
    "genre_fusion": 0.6,

    "clarity": 0.7,
    "feature_consistency": 0.8
}

agents = [

    MelodyHarmonyAgent(env),
    RhythmGrooveAgent(env),
    StructureAgent(env),
    EmotionAgent(env),
    InnovationAgent(env),
    ExplainabilityAgent(env)

]

evaluations = []

for agent in agents:
    result = agent.evaluate(music_features)
    print(result)
    evaluations.append(result)


judge = JudgeAgent(env)

final = judge.evaluate(evaluations)

print("\nFinal Evaluation:")
print(final)