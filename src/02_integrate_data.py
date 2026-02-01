# 02_integrate_data.py

import pandas as pd
import glob
import os
import numpy as np

# 경로
CHANNEL_PATH			= "../data/config/channel.txt"
RAW_PATH				= "../data/raw"
OUTPUT_PATH				= "../data/data/integrated_data.parquet"

# 모든 채널 폴더 내의 엑셀 파일을 찾습니다.
all_files = glob.glob(os.path.join(RAW_PATH, "*/*.xlsx"))
all_data = []

print(f"--- 데이터 통합 시작 (총 {len(all_files)}개 파일) ---")

for i, file in enumerate(all_files):
	try:
		video_title = os.path.splitext(os.path.basename(file))[0]
		channel_name = os.path.basename(os.path.dirname(file)).strip()

		# 시트별 데이터
		video_df = pd.read_excel(file, sheet_name='Video_Stats')
		comments_df = pd.read_excel(file, sheet_name='All_Comments')

		if comments_df.empty:
			continue

		comment_time = pd.to_datetime(comments_df['Published_At'])
		video_time = pd.to_datetime(video_df['Published_At'].iloc[0])
		time_diffs = (comment_time - video_time).dt.total_seconds() / 60

		df = pd.DataFrame({
			'comment': comments_df['Comment'],				# 댓글 내용
			'likes': comments_df['Likes'],					# 댓글 좋아요 수
			'video_title': video_title,						# 영상 제목
			'channel_name': channel_name,					# 채널명
			'video_views': video_df['Views'].iloc[0],		# 영상 조회수
			'time_diff': time_diffs.clip(lower=0)			# 댓글 게시시간 - 영상 게시시간
		})

		all_data.append(df)

		if (i + 1) % 1000 == 0:
			print(f"{i + 1}개 파일 처리 완료")

	except Exception as e:
		print(f"파일 처리 에러 ({file}): {e}")
		continue

# 모든 데이터 통합
if all_data:
	channel_df = pd.read_csv(CHANNEL_PATH, header=None, names=['channel_id', 'channel_name', 'subscribers', 'total_videos'], encoding='utf-8')
	channel_df['channel_name'] = channel_df['channel_name'].str.strip()

	total_data = pd.concat(all_data, ignore_index=True)

	total_data = total_data.merge(channel_df[['channel_name', 'subscribers']], on='channel_name', how='left')
	total_data = total_data.dropna(subset=['comment', 'likes', 'subscribers'])

	total_data.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)
	print(f"\n통합 완료! 총 {len(total_data):,}개의 댓글 데이터가 저장되었습니다.")
else:
	print("통합할 데이터가 없습니다.")