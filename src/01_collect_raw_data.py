# 01_collect_raw_data.py

import re
import os
import pandas as pd
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import Counter

# 경로
CHANNEL_PATH			= "../data/config/channel.txt"
RAW_PATH				= "../data/raw"

def load_api_keys():
	with open("data/config/api_key.txt", "r") as f:
		return [line.strip() for line in f if line.strip()]

API_KEYS = load_api_keys()
KEY_INDEX = 0

def get_youtube_client():
	global KEY_INDEX
	return build('youtube', 'v3', developerKey=API_KEYS[KEY_INDEX], static_discovery=False)

YOUTUBE = get_youtube_client()

def switch_api_key(error_message):
	global KEY_INDEX, YOUTUBE
	# 댓글 중지 에러
	if "commentsDisabled" in error_message:
		return True

	print(f"\n할당량 소진 감지 ({KEY_INDEX+1}번 키) -> 다음 키로 교체 시도")
	KEY_INDEX += 1
	if KEY_INDEX < len(API_KEYS):
		print(f"{KEY_INDEX+1}번째 키로 교체합니다...")
		time.sleep(1)
		try:
			YOUTUBE = get_youtube_client()
			return True
		except Exception:
			return switch_api_key("연속 교체")
	return False

def clean_filename(filename):
	return re.sub(r'[\\/*?:"<>|]', "", filename)

# 영상 리스트 획득
def get_video_list_with_titles(c_id):
	global YOUTUBE
	video_data = []
	try:
		ch_resp = YOUTUBE.channels().list(part='contentDetails', id=c_id).execute()
		if not ch_resp.get('items'): return []
		upload_id = ch_resp['items'][0]['contentDetails']['relatedPlaylists']['uploads']
		next_pt = None
		while True:
			try:
				res = YOUTUBE.playlistItems().list(playlistId=upload_id, part='snippet', maxResults=50, pageToken=next_pt).execute()
				for item in res['items']:
					v_id = item['snippet']['resourceId']['videoId']
					v_title = item['snippet']['title']
					if v_title not in ['Private video', 'Deleted video']:
						video_data.append({'id': v_id, 'title': clean_filename(v_title)})
				next_pt = res.get('nextPageToken')
				if not next_pt: break
			except HttpError as e:
				if "quotaExceeded" in str(e) or e.resp.status == 403:
					if switch_api_key(str(e)): continue
				break
		return video_data
	except: return []

# 개별 영상 수집
def create_video_report(video_id, final_filename, channel_name):
	global YOUTUBE
	while True:
		try:
			channel_folder = os.path.join(RAW_PATH, channel_name)
			file_path = os.path.join(channel_folder, f"{final_filename}.xlsx")
			if os.path.exists(file_path):
				print(f"건너뛰기: {final_filename}")
				return

			v_resp = YOUTUBE.videos().list(part='snippet,statistics', id=video_id).execute()
			if not v_resp['items']: return
			
			v_item = v_resp['items'][0]
			snippet = v_item['snippet']
			stats = v_item['statistics']
			pub_at = pd.to_datetime(snippet['publishedAt']).tz_localize(None)

			all_comments = []
			total_comment_likes = 0
			next_page_token = None

			while True:
				try:
					c_resp = YOUTUBE.commentThreads().list(
						part='snippet', videoId=video_id, order='relevance', 
						maxResults=100, pageToken=next_page_token
					).execute()

					for item in c_resp['items']:
						node = item['snippet']['topLevelComment']['snippet']
						likes = int(node.get('likeCount', 0))
						total_comment_likes += likes
						all_comments.append({
							'Author': node['authorDisplayName'], 'Comment': node['textDisplay'],
							'Likes': likes, 'Published_At': pd.to_datetime(node['publishedAt']).tz_localize(None)
						})
					next_page_token = c_resp.get('nextPageToken')
					if not next_page_token: break
				except HttpError as e:
					# 댓글 중지 여부 확인
					if "commentsDisabled" in str(e):
						print(f" (댓글 중지)", end="")
						break
					if e.resp.status == 403:
						if switch_api_key(str(e)): raise Exception("RetryWithNewKey")
						else: exit()
					break

			df_stats = pd.DataFrame([{
				'Video_ID': video_id, 'Title': snippet['title'],
				'Views': int(stats.get('viewCount', 0)), 'Likes': int(stats.get('likeCount', 0)),
				'Total_Comments': int(stats.get('commentCount', 0)),
				'Total_Comment_Likes': total_comment_likes, 'Published_At': pub_at
			}])
			df_comments = pd.DataFrame(all_comments)
			df_best = df_comments.sort_values(by='Likes', ascending=False).head(1) if not df_comments.empty else pd.DataFrame()

			os.makedirs(channel_folder, exist_ok=True)
			with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
				df_stats.to_excel(writer, sheet_name='Video_Stats', index=False)
				df_comments.to_excel(writer, sheet_name='All_Comments', index=False)
				df_best.to_excel(writer, sheet_name='Best_Comments', index=False)

			print(f"성공: {final_filename} (댓글 {len(all_comments)}개)")
			break
		except Exception as e:
			if "RetryWithNewKey" in str(e): continue
			print(f"에러: {e}"); break

# 채널의 전체 영상 개수만 빠르게 가져오는 함수
def get_channel_video_count(c_id):
	global YOUTUBE
	while True: # 키 교체 후 재시도를 위해 루프 추가
		try:
			resp = YOUTUBE.channels().list(part='statistics', id=c_id).execute()
			if resp.get('items'):
				return int(resp['items'][0]['statistics']['videoCount'])
			return 0
		except HttpError as e:
			# 할당량 초과 시 키 교체
			if e.resp.status == 403:
				if switch_api_key(str(e)):
					continue # 새로운 키로 다시 시도
				else:
					exit() # 키가 더 없으면 종료
			return 0
		except Exception as e:
			print(f" (기타 오류: {e})")
			return 0

# 메인
if __name__ == "__main__":
	CANDIDATE_CHANNELS = []
	if os.path.exists(CHANNEL_PATH):
		with open(CHANNEL_PATH, "r", encoding="utf-8") as f:
			for line in f:
				parts = line.strip().split(",")
				if len(parts) >= 2: CANDIDATE_CHANNELS.append({"id": parts[0], "name": parts[1]})

	while True:
		user_input = input("\n명령어(all / recommend / exit): ").strip().lower()
		if user_input == 'exit': break
		target_list = CANDIDATE_CHANNELS if user_input == 'all' else []
		if user_input == 'recommend' and CANDIDATE_CHANNELS:
			for ch in CANDIDATE_CHANNELS:
				if not os.path.exists(os.path.join(RAW_PATH, clean_filename(ch['name']))):
					target_list = [ch]; break

		for ch_info in target_list:
			ch_name_clean = clean_filename(ch_info['name'])
			ch_folder = os.path.join(RAW_PATH, ch_name_clean)

			# 스킵 체크
			print(f"[{ch_info['name']}] 스킵 여부 확인 중...", end="")

			# 내 폴더의 파일 개수와, 채널의 전체 영상 개수 비교 (비공개 포함)
			existing_count = len([f for f in os.listdir(ch_folder) if f.endswith('.xlsx')]) if os.path.exists(ch_folder) else 0
			stat_video_count = get_channel_video_count(ch_info['id'])

			# 정밀 검사 없이 스킵
			if stat_video_count > 0 and existing_count >= stat_video_count:
				print(f" -> 완료 스킵 (파일 {existing_count}개 / 통계 {stat_video_count}개)")
				continue

			# 정밀 검사 (파일 갯수가 부족한 경우)
			video_list = get_video_list_with_titles(ch_info['id'])
			total = len(video_list)

			if total == 0: continue

			# 실제 유효 영상 기준으로 다시 한번 스킵 확인
			if existing_count >= total:
				print(f" -> 정밀 검사 결과 완료 스킵 ({existing_count}개 / 통계 {total}개)")
				continue

			# 영상 정보 수집
			title_counts = Counter([v['title'] for v in video_list])
			tracker = {}
			for i, v in enumerate(video_list, 1):
				raw_title = v['title']
				if title_counts[raw_title] > 1:
					tracker[raw_title] = tracker.get(raw_title, 0) + 1
					final_name = f"{raw_title} ({tracker[raw_title]})"
				else: final_name = raw_title

				# 건너뛰기는 create_video_report 안에서 체크
				print(f"[{i}/{total}]", end=" ")
				create_video_report(v['id'], final_name, ch_name_clean)