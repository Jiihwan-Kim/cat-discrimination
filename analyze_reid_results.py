import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def load_reid_results(results_dir):
    """ReID 결과 로드"""
    results = []
    
    for json_file in Path(results_dir).glob('*_results.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            results.append(json.load(f))
    
    return results

def analyze_tracks(results):
    """Track 분석"""
    track_stats = []
    
    for video_result in results:
        video_name = Path(video_result['video_path']).stem
        
        for track_id, track_info in video_result['tracks'].items():
            track_duration = track_info['last_frame'] - track_info['first_frame']
            num_detections = len(track_info['detections'])
            
            track_stats.append({
                'video': video_name,
                'track_id': int(track_id),  # 문자열을 정수로 변환
                'duration': track_duration,
                'detections': num_detections,
                'first_frame': track_info['first_frame'],
                'last_frame': track_info['last_frame']
            })
    
    return pd.DataFrame(track_stats)

def visualize_track_analysis(df, output_dir):
    """Track 분석 시각화"""
    plt.figure(figsize=(15, 10))
    
    # 1. Track 지속 시간 분포
    plt.subplot(2, 3, 1)
    plt.hist(df['duration'], bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Track Duration (frames)')
    plt.ylabel('Count')
    plt.title('Track Duration Distribution')
    
    # 2. 감지 횟수 분포
    plt.subplot(2, 3, 2)
    plt.hist(df['detections'], bins=20, alpha=0.7, color='lightgreen')
    plt.xlabel('Number of Detections')
    plt.ylabel('Count')
    plt.title('Detection Count Distribution')
    
    # 3. 비디오별 track 수
    plt.subplot(2, 3, 3)
    video_track_counts = df.groupby('video')['track_id'].nunique()
    plt.bar(range(len(video_track_counts)), video_track_counts.values)
    plt.xlabel('Video Index')
    plt.ylabel('Number of Tracks')
    plt.title('Tracks per Video')
    
    # 4. Track 지속 시간 vs 감지 횟수
    plt.subplot(2, 3, 4)
    plt.scatter(df['duration'], df['detections'], alpha=0.6)
    plt.xlabel('Track Duration (frames)')
    plt.ylabel('Number of Detections')
    plt.title('Duration vs Detections')
    
    # 5. 비디오별 평균 track 지속 시간
    plt.subplot(2, 3, 5)
    avg_duration = df.groupby('video')['duration'].mean()
    plt.bar(range(len(avg_duration)), avg_duration.values)
    plt.xlabel('Video Index')
    plt.ylabel('Average Track Duration')
    plt.title('Average Track Duration per Video')
    
    # 6. Track ID 분포
    plt.subplot(2, 3, 6)
    track_id_counts = df['track_id'].value_counts()
    plt.hist(track_id_counts.values, bins=10, alpha=0.7, color='orange')
    plt.xlabel('Track ID Frequency')
    plt.ylabel('Count')
    plt.title('Track ID Frequency Distribution')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'track_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_track_timeline(results, output_dir):
    """Track 타임라인 생성"""
    plt.figure(figsize=(20, 10))
    
    colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    for i, video_result in enumerate(results):
        video_name = Path(video_result['video_path']).stem
        
        for track_id, track_info in video_result['tracks'].items():
            start_frame = track_info['first_frame']
            end_frame = track_info['last_frame']
            
            # track_id를 정수로 변환
            track_id_int = int(track_id)
            
            plt.plot([start_frame, end_frame], [i, i], 
                    color=colors[track_id_int % len(colors)], 
                    linewidth=3, 
                    label=f'Track {track_id_int}' if i == 0 else "")
    
    plt.xlabel('Frame Number')
    plt.ylabel('Video Index')
    plt.title('Track Timeline Across Videos')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'track_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='ReID 결과 분석')
    parser.add_argument('--results_dir', type=str, default='reid_output',
                       help='ReID 결과 디렉토리')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='분석 결과 출력 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # 결과 로드
    results = load_reid_results(args.results_dir)
    print(f"로드된 결과 파일 수: {len(results)}")
    
    # Track 분석
    df = analyze_tracks(results)
    print(f"총 track 수: {len(df)}")
    print(f"고유한 track ID 수: {df['track_id'].nunique()}")
    
    # 통계 출력
    print("\n=== Track 통계 ===")
    print(f"평균 track 지속 시간: {df['duration'].mean():.1f} 프레임")
    print(f"평균 감지 횟수: {df['detections'].mean():.1f}회")
    print(f"최장 track 지속 시간: {df['duration'].max()} 프레임")
    print(f"최대 감지 횟수: {df['detections'].max()}회")
    
    # 시각화
    visualize_track_analysis(df, args.output_dir)
    generate_track_timeline(results, args.output_dir)
    
    # 상세 통계 저장
    stats_summary = {
        'total_tracks': len(df),
        'unique_track_ids': df['track_id'].nunique(),
        'total_videos': len(results),
        'avg_duration': df['duration'].mean(),
        'avg_detections': df['detections'].mean(),
        'max_duration': df['duration'].max(),
        'max_detections': df['detections'].max(),
        'track_stats': df.to_dict('records')
    }
    
    with open(Path(args.output_dir) / 'analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n분석 완료! 결과 저장 위치: {args.output_dir}")

if __name__ == "__main__":
    main() 