from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional

from agent.llm.ollama_client import OllamaClient
from agent.config import get_settings
from agent.services.teaching import TeachingService
from agent.azl.generator import AZLGenerator
from agent.services.validators import AZLValidators
from agent.voice.stt import WhisperSTT, STT_AVAILABLE
from agent.voice.tts import ChatterboxTTS, TTS_AVAILABLE

# Check if we're in an interactive terminal
IS_INTERACTIVE = sys.stdout.isatty()

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m' if IS_INTERACTIVE else ''
    BLUE = '\033[94m' if IS_INTERACTIVE else ''
    CYAN = '\033[96m' if IS_INTERACTIVE else ''
    GREEN = '\033[92m' if IS_INTERACTIVE else ''
    WARNING = '\033[93m' if IS_INTERACTIVE else ''
    FAIL = '\033[91m' if IS_INTERACTIVE else ''
    ENDC = '\033[0m' if IS_INTERACTIVE else ''
    BOLD = '\033[1m' if IS_INTERACTIVE else ''
    UNDERLINE = '\033[4m' if IS_INTERACTIVE else ''

def play_audio_bytes(audio_data: bytes, sample_rate: int = 22050) -> None:
    """Play PCM16 audio bytes if optional deps are available.
    Falls back gracefully when missing (avoids hard dependency and pyright errors).
    """
    try:
        import importlib
        np = importlib.import_module('numpy')
        sd = importlib.import_module('sounddevice')
        audio_array = np.frombuffer(audio_data, dtype=np.int16)  # type: ignore[attr-defined]
        sd.play(audio_array, sample_rate)  # type: ignore[attr-defined]
        sd.wait()  # type: ignore[attr-defined]
    except Exception as e:
        print(f"{Colors.WARNING}Audio playback skipped (missing deps): {e}{Colors.ENDC}")

async def main() -> None:
    parser = argparse.ArgumentParser(description="MentorZero CLI demo")
    parser.add_argument("topic", help="Topic to teach")
    parser.add_argument("--mode", default="explain", choices=["explain", "quiz"], 
                        help="Teaching mode: explain concepts or quiz knowledge")
    parser.add_argument("--difficulty", default="beginner", 
                        choices=["beginner", "intermediate", "advanced", "expert"],
                        help="Difficulty level for content")
    parser.add_argument("--strategy", default="neural_compression",
                        choices=["neural_compression", "socratic_dialogue", "concrete_examples", 
                                "visual_mapping", "spaced_repetition"],
                        help="Teaching strategy to use")
    
    # AZL options
    azl_group = parser.add_argument_group('AZL (Absolute Zero Learning)')
    azl_group.add_argument("--azl", action="store_true", help="Enable AZL features")
    azl_group.add_argument("--azl-propose", action="store_true", 
                          help="Propose synthetic examples for the topic")
    azl_group.add_argument("--azl-validate", type=str, metavar="PROPOSAL_FILE",
                          help="Validate examples from a proposal file")
    azl_group.add_argument("--azl-report", action="store_true",
                          help="Show report of validated examples")
    
    # Voice options
    voice_group = parser.add_argument_group('Voice')
    voice_group.add_argument("--voice", action="store_true", 
                            help="Enable voice output (requires TTS)")
    voice_group.add_argument("--voice-input", action="store_true",
                            help="Enable voice input (requires STT)")
    voice_group.add_argument("--voice-model", type=str, default="default",
                            help="Voice model to use for TTS")
    
    args = parser.parse_args()

    # Setup clients
    s = get_settings()
    llm = OllamaClient(s.ollama_host, s.ollama_model, timeout_seconds=(s.ollama_timeout_seconds or 60.0))
    ts = TeachingService(llm)
    
    # Initialize session
    session_id = f"cli-{os.getpid()}"
    user_id = "cli-user"
    
    # Initialize mastery tracking
    ts._initialize_mastery_tracking(session_id, user_id, args.topic)
    
    # Override strategy if specified
    if args.strategy:
        tracking = ts._mastery_tracking.get(session_id, {})
        if tracking:
            tracking["current_strategy"] = args.strategy
            tracking["current_difficulty"] = args.difficulty
    
    # Handle AZL commands
    if args.azl:
        if not await check_llm_health(llm):
            print(f"{Colors.FAIL}Error: LLM is not available. Please check your Ollama server.{Colors.ENDC}")
            return
            
        if args.azl_propose:
            await handle_azl_propose(llm, args.topic, args.difficulty)
            return
        elif args.azl_validate:
            await handle_azl_validate(llm, args.azl_validate)
            return
        elif args.azl_report:
            handle_azl_report()
            return
    
    # Check LLM health
    if not await check_llm_health(llm):
        print(f"{Colors.FAIL}Error: LLM is not available. Please check your Ollama server.{Colors.ENDC}")
        return
    
    # Setup voice if requested
    tts = None
    if args.voice:
        if not TTS_AVAILABLE:
            print(f"{Colors.WARNING}Warning: TTS is not available. Install chatterbox-tts package.{Colors.ENDC}")
        else:
            tts = ChatterboxTTS()
    
    stt = None
    if args.voice_input:
        if not STT_AVAILABLE:
            print(f"{Colors.WARNING}Warning: STT is not available. Install openai-whisper package.{Colors.ENDC}")
        else:
            stt = WhisperSTT()
    
    # Main teaching flow
    if args.mode == "explain":
        print(f"{Colors.HEADER}Generating explanation for: {Colors.BOLD}{args.topic}{Colors.ENDC}")
        print(f"{Colors.CYAN}Difficulty: {args.difficulty}, Strategy: {args.strategy}{Colors.ENDC}")
        print("Please wait...")
        
        explanation = await ts.generate_explanation(
            llm, 
            user_id, 
            args.topic, 
            args.mode,
            session_id,
            None
        )
        
        print(f"\n{Colors.GREEN}{explanation}{Colors.ENDC}")
        
        # Voice output if requested
        if args.voice and tts and TTS_AVAILABLE:
            print(f"{Colors.CYAN}Generating voice...{Colors.ENDC}")
            audio_data = await tts.synthesize(explanation)
            if audio_data:
                play_audio_bytes(audio_data, 22050)
        
        # Interactive loop for follow-up questions
        if IS_INTERACTIVE:
            print(f"\n{Colors.HEADER}You can ask follow-up questions (or type 'exit' to quit):{Colors.ENDC}")
            
            while True:
                if args.voice_input and stt and STT_AVAILABLE:
                    print(f"{Colors.CYAN}Listening... (speak your question){Colors.ENDC}")
                    user_input = await record_and_transcribe(stt)
                    print(f"You said: {user_input}")
                else:
                    user_input = input(f"{Colors.BOLD}> {Colors.ENDC}")
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                
                print(f"{Colors.CYAN}Generating response...{Colors.ENDC}")
                
                # Get feedback using the submit_answer flow
                feedback, _ = await ts.generate_feedback(
                    llm,
                    session_id,
                    user_input,
                    None
                )
                
                print(f"\n{Colors.GREEN}{feedback}{Colors.ENDC}")
                
                # Voice output if requested
                if args.voice and tts and TTS_AVAILABLE:
                    audio_data = await tts.synthesize(feedback)
                    if audio_data:
                        play_audio_bytes(audio_data, 22050)
    else:
        # Quiz mode
        print(f"{Colors.HEADER}Generating quiz for: {Colors.BOLD}{args.topic}{Colors.ENDC}")
        print(f"{Colors.CYAN}Difficulty: {args.difficulty}, Strategy: {args.strategy}{Colors.ENDC}")
        print("Please wait...")
        
        quiz = await ts.generate_quiz(
            llm,
            args.topic,
            args.difficulty,
            None
        )
        
        if not quiz:
            print(f"{Colors.FAIL}Error: Failed to generate quiz.{Colors.ENDC}")
            return
        
        # Interactive quiz
        correct = 0
        for i, q in enumerate(quiz, 1):
            print(f"\n{Colors.HEADER}Question {i}/{len(quiz)}{Colors.ENDC}")
            print(f"{Colors.BOLD}{q['question']}{Colors.ENDC}")
            
            if 'options' in q and len(q['options']) > 0:
                # Multiple choice
                options = q['options'] + [q['answer']]
                import random
                random.shuffle(options)
                
                for j, opt in enumerate(options):
                    print(f"{j+1}. {opt}")
                
                # Voice output if requested
                if args.voice and tts and TTS_AVAILABLE:
                    question_text = f"Question {i}. {q['question']}. "
                    for j, opt in enumerate(options):
                        question_text += f"Option {j+1}: {opt}. "
                    
                    audio_data = await tts.synthesize(question_text)
                    if audio_data:
                        play_audio_bytes(audio_data, 22050)
                
                # Get user answer
                if args.voice_input and stt and STT_AVAILABLE:
                    print(f"{Colors.CYAN}Listening... (speak your answer){Colors.ENDC}")
                    user_input = await record_and_transcribe(stt)
                    print(f"You said: {user_input}")
                    
                    # Try to extract a number
                    import re
                    match = re.search(r'\b([0-9]|one|two|three|four|five)\b', user_input.lower())
                    if match:
                        num_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
                        try:
                            choice = int(num_map.get(match.group(1), match.group(1)))
                            if 1 <= choice <= len(options):
                                user_answer = options[choice-1]
                            else:
                                user_answer = user_input
                        except:
                            user_answer = user_input
                    else:
                        user_answer = user_input
                else:
                    user_input = input(f"{Colors.BOLD}Your answer (1-{len(options)}): {Colors.ENDC}")
                    try:
                        choice = int(user_input)
                        if 1 <= choice <= len(options):
                            user_answer = options[choice-1]
                        else:
                            user_answer = user_input
                    except:
                        user_answer = user_input
            else:
                # Free-form answer
                if args.voice_input and stt and STT_AVAILABLE:
                    print(f"{Colors.CYAN}Listening... (speak your answer){Colors.ENDC}")
                    user_answer = await record_and_transcribe(stt)
                    print(f"You said: {user_answer}")
                else:
                    user_answer = input(f"{Colors.BOLD}Your answer: {Colors.ENDC}")
            
            # Check answer
            feedback, is_correct = await ts.generate_feedback(
                llm,
                session_id,
                user_answer,
                None
            )
            
            if is_correct:
                print(f"{Colors.GREEN}Correct! {Colors.ENDC}")
                correct += 1
            else:
                print(f"{Colors.WARNING}Not quite. {Colors.ENDC}")
            
            print(f"{Colors.CYAN}Feedback: {feedback}{Colors.ENDC}")
            print(f"{Colors.BOLD}Correct answer: {q['answer']}{Colors.ENDC}")
            
            # Voice output if requested
            if args.voice and tts and TTS_AVAILABLE:
                result = "Correct!" if is_correct else "Not quite."
                feedback_text = f"{result} {feedback} The correct answer is: {q['answer']}"
                
                audio_data = await tts.synthesize(feedback_text)
                if audio_data:
                    import sounddevice as sd
                    import numpy as np
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    sd.play(audio_array, 22050)
                    sd.wait()
        
        # Quiz summary
        print(f"\n{Colors.HEADER}Quiz complete!{Colors.ENDC}")
        print(f"You got {correct}/{len(quiz)} correct ({int(correct/len(quiz)*100)}%)")
        
        # Update mastery tracking
        tracking = ts._mastery_tracking.get(session_id, {})
        if tracking:
            print(f"\n{Colors.HEADER}Learning Progress:{Colors.ENDC}")
            print(f"Mastery Level: {tracking.get('mastery_level', 0)}/10")
            print(f"Current Difficulty: {tracking.get('current_difficulty', 'beginner')}")
            print(f"Current Strategy: {tracking.get('current_strategy', 'neural_compression')}")
            print(f"Streak: {tracking.get('current_streak', 0)}")

async def check_llm_health(llm: OllamaClient) -> bool:
    """Check if the LLM is available and ready"""
    try:
        await llm.health_check()
        return True
    except Exception:
        return False

async def handle_azl_propose(llm: OllamaClient, topic: str, difficulty: str) -> None:
    """Handle AZL proposal generation"""
    print(f"{Colors.HEADER}Generating AZL proposals for: {Colors.BOLD}{topic}{Colors.ENDC}")
    print(f"{Colors.CYAN}Difficulty: {difficulty}{Colors.ENDC}")
    print("Please wait...")
    
    azl_generator = AZLGenerator(llm)
    examples = await azl_generator.propose_examples(topic, 5)
    
    if not examples:
        print(f"{Colors.FAIL}Error: Failed to generate examples.{Colors.ENDC}")
        return
    
    # Save to file
    filename = f"azl_proposal_{topic.replace(' ', '_')}.json"
    with open(filename, 'w') as f:
        json.dump({"topic": topic, "difficulty": difficulty, "examples": examples}, f, indent=2)
    
    print(f"{Colors.GREEN}Generated {len(examples)} examples and saved to {filename}{Colors.ENDC}")
    
    # Preview
    print(f"\n{Colors.HEADER}Example Preview:{Colors.ENDC}")
    for i, example in enumerate(examples[:2], 1):
        print(f"\n{Colors.BOLD}Example {i}:{Colors.ENDC}")
        print(f"Q: {example.get('question', '')}")
        print(f"A: {example.get('answer', '')}")

async def handle_azl_validate(llm: OllamaClient, proposal_file: str) -> None:
    """Handle AZL validation"""
    print(f"{Colors.HEADER}Validating examples from: {Colors.BOLD}{proposal_file}{Colors.ENDC}")
    
    try:
        with open(proposal_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"{Colors.FAIL}Error: Failed to read proposal file: {e}{Colors.ENDC}")
        return
    
    validators = AZLValidators(llm)
    results = []
    
    for i, example in enumerate(data.get("examples", []), 1):
        print(f"\n{Colors.BOLD}Validating Example {i}/{len(data.get('examples', []))}:{Colors.ENDC}")
        print(f"Q: {example.get('question', '')}")
        print(f"A: {example.get('answer', '')}")
        
        # Run validations
        length_check = await validators.validate_length(example)
        duplicate_check = await validators.validate_no_duplicates(example)
        consistency_check = await validators.validate_consistency(example)
        roundtrip_check = await validators.validate_roundtrip(example)
        toxicity_check = await validators.validate_no_toxicity(example)
        
        # Collect results
        validation_result = {
            "example": example,
            "validations": {
                "length": length_check,
                "duplicate": duplicate_check,
                "consistency": consistency_check,
                "roundtrip": roundtrip_check,
                "toxicity": toxicity_check
            },
            "passed": all([
                length_check.get("passed", False),
                duplicate_check.get("passed", False),
                consistency_check.get("passed", False),
                roundtrip_check.get("passed", False),
                toxicity_check.get("passed", False)
            ])
        }
        results.append(validation_result)
        
        # Display results
        for check_name, check_result in validation_result["validations"].items():
            status = f"{Colors.GREEN}PASS{Colors.ENDC}" if check_result.get("passed", False) else f"{Colors.FAIL}FAIL{Colors.ENDC}"
            print(f"  {check_name}: {status} - {check_result.get('message', '')}")
    
    # Save validation results
    output_file = f"{os.path.splitext(proposal_file)[0]}_validated.json"
    with open(output_file, 'w') as f:
        json.dump({"topic": data.get("topic", ""), "results": results}, f, indent=2)
    
    print(f"\n{Colors.GREEN}Validation complete. Results saved to {output_file}{Colors.ENDC}")
    
    # Summary
    passed = sum(1 for r in results if r.get("passed", False))
    print(f"{Colors.HEADER}Summary: {passed}/{len(results)} examples passed all validations{Colors.ENDC}")

def handle_azl_report() -> None:
    """Show report of validated examples"""
    print(f"{Colors.HEADER}AZL Validation Report{Colors.ENDC}")
    
    # Find all validation files
    import glob
    validation_files = glob.glob("*_validated.json")
    
    if not validation_files:
        print(f"{Colors.WARNING}No validation files found.{Colors.ENDC}")
        return
    
    total_examples = 0
    total_passed = 0
    
    for file in validation_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            topic = data.get("topic", "Unknown")
            results = data.get("results", [])
            passed = sum(1 for r in results if r.get("passed", False))
            
            print(f"\n{Colors.BOLD}Topic: {topic}{Colors.ENDC}")
            print(f"File: {file}")
            print(f"Examples: {len(results)}, Passed: {passed} ({int(passed/max(1,len(results))*100)}%)")
            
            total_examples += len(results)
            total_passed += passed
        except Exception as e:
            print(f"{Colors.FAIL}Error reading {file}: {e}{Colors.ENDC}")
    
    if total_examples > 0:
        print(f"\n{Colors.HEADER}Overall: {total_passed}/{total_examples} examples passed ({int(total_passed/total_examples*100)}%){Colors.ENDC}")

async def record_and_transcribe(stt: WhisperSTT) -> str:
    """Record audio and transcribe it"""
    try:
        import importlib
        np = importlib.import_module('numpy')
        sd = importlib.import_module('sounddevice')

        # Record audio
        duration = 5  # seconds
        fs = 16000  # sample rate
        print(f"{Colors.CYAN}Recording for {duration} seconds...{Colors.ENDC}")
        
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')  # type: ignore[attr-defined]
        sd.wait()  # type: ignore[attr-defined]
        
        # Convert to bytes
        audio_data = recording.tobytes()
        
        # Transcribe
        transcript = await stt.transcribe(audio_data)
        return transcript
    except Exception as e:
        print(f"{Colors.FAIL}Error recording/transcribing: {e}{Colors.ENDC}")
        return ""

if __name__ == "__main__":
    asyncio.run(main())