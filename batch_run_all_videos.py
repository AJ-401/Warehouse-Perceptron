"""
batch_run_all_videos.py
=======================
Batch runner for Person A Pipeline across all 7 official Godrej challenge videos.
Loads the perception models once into memory and processes each video sequentially.
"""

import glob
import json
import os
import time
from pipeline_person_a import PersonAPipeline


def run_batch():
    input_dir = "official_videos"
    output_dir = "outputs_person_a"
    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    if not video_files:
        print(f"No .mp4 files found in {input_dir}")
        return

    print("=" * 70)
    print(f"PERSON A PIPELINE: BATCH PROCESSING {len(video_files)} VIDEOS")
    print(f"Input Directory : {input_dir}")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)

    # Initialize pipeline once
    pipeline = PersonAPipeline(box_conf=0.15, person_conf=0.35)

    summary_results = []
    total_batch_start = time.time()

    for idx, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        base_name = os.path.splitext(filename)[0]
        json_path = os.path.join(output_dir, f"{base_name}_tracking_results.json")
        mp4_path = os.path.join(output_dir, f"{base_name}_perception.mp4")

        # Check if already processed completely
        if os.path.exists(json_path) and os.path.exists(mp4_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    meta = data.get("video_metadata", {})
                    if meta.get("total_frames_processed", 0) > 0:
                        print(f"\n[{idx}/{len(video_files)}] ALREADY PROCESSED: {filename} ({meta.get('total_frames_processed')} frames, {meta.get('average_fps')} FPS)")
                        summary_results.append({
                            "video": filename,
                            "frames": meta.get("total_frames_processed"),
                            "time_sec": meta.get("processing_time_sec"),
                            "fps": meta.get("average_fps"),
                            "status": "Cached"
                        })
                        continue
            except Exception:
                pass

        print(f"\n[{idx}/{len(video_files)}] STARTING: {filename}")
        t0 = time.time()
        try:
            pipeline.run(
                video_path=video_path,
                output_dir=output_dir,
                max_frames=None,  # Full video
                save_video=True
            )
            elapsed = time.time() - t0

            # Read back metadata
            with open(json_path, "r") as f:
                meta = json.load(f).get("video_metadata", {})

            summary_results.append({
                "video": filename,
                "frames": meta.get("total_frames_processed", 0),
                "time_sec": meta.get("processing_time_sec", round(elapsed, 2)),
                "fps": meta.get("average_fps", 0),
                "status": "Success"
            })
        except Exception as e:
            print(f"ERROR processing {filename}: {e}")
            summary_results.append({
                "video": filename,
                "frames": 0,
                "time_sec": round(time.time() - t0, 2),
                "fps": 0,
                "status": f"Failed: {e}"
            })

    total_batch_time = time.time() - total_batch_start
    print("\n" + "=" * 80)
    print(f"BATCH PROCESSING COMPLETE IN {total_batch_time:.2f}s ({total_batch_time/60:.1f} min)")
    print("=" * 80)
    print(f"{'Video Name':<50} | {'Frames':<8} | {'Time (s)':<10} | {'FPS':<6} | {'Status'}")
    print("-" * 80)
    for r in summary_results:
        print(f"{r['video'][:48]:<50} | {r['frames']:<8} | {r['time_sec']:<10} | {r['fps']:<6} | {r['status']}")
    print("=" * 80)

    # Save summary report
    summary_path = os.path.join(output_dir, "batch_processing_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total_videos": len(video_files),
            "total_time_sec": round(total_batch_time, 2),
            "results": summary_results
        }, f, indent=2)
    print(f"Summary report saved to: {summary_path}")


if __name__ == "__main__":
    run_batch()
