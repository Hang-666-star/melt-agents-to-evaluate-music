import sys
from extract_features import extract_music_features
from marble.environments.music_env import MusicEvaluationEnvironment
from marble.agent.music_agent import (
    MelodyHarmonyAgent,
    RhythmGrooveAgent,
    StructureAgent,
    EmotionAgent,
    InnovationAgent,
    ExplainabilityAgent,
    JudgeAgent,
    DiscussionOrchestrator
)
from music_report import generate_music_report


def evaluate(audio_file):
    print("========== MUSIC EVALUATION SYSTEM ==========")
    print("\n part1: Extracting audio features...")
    features = extract_music_features(audio_file)
    print("\nExtracted Features:")
    print(features)

    print("\n part2: Initializing environment...")
    env = MusicEvaluationEnvironment(features)

    print("\n part3: Creating expert agents...")
    llm_config = {
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-bde5f512f8b346e2b9ea0129d2538016",   
        "model_name": "deepseek-chat"
    }
    agents = [
        MelodyHarmonyAgent(env, llm_config),
        RhythmGrooveAgent(env, llm_config),
        StructureAgent(env, llm_config),
        EmotionAgent(env, llm_config),
        InnovationAgent(env, llm_config),
        ExplainabilityAgent(env, llm_config)
    ]

    print("\n part4: Starting multi-round discussion...")
    orchestrator = DiscussionOrchestrator(agents, max_round=3)
    final_round_results = orchestrator.run(features)

    print("\n part5: Aggregating final scores...")
    judge = JudgeAgent(env)
    final = judge.evaluate(final_round_results)
    print("\nFinal Evaluation Result:")
    print(final)

    print("\n part6: Generating AI music report...")
    report = generate_music_report(final_round_results, final["final_score"])
    print("\n========== MUSIC EVALUATION REPORT ==========\n")
    print(report)
    print("\n=============================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_music.py <audio_file>")
        sys.exit(1)
    audio_path = sys.argv[1]
    evaluate(audio_path)