import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
from google import genai

# --- 1. 기본 설정 ---

# .env 파일 로드
load_dotenv()

# Flask 앱 초기화
app = Flask(__name__, static_folder='img')
CORS(app) # CORS 설정 추가

# Gemini API 클라이언트 초기화
try:
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
    #model = genai.GenerativeModel('gemini-2.0-flash-preview-image-generation') # 모델 이름은 최신 버전으로 변경될 수 있습니다.
except Exception as e:
    print(f"API 키 설정에 실패했습니다. .env 파일을 확인해주세요: {e}")
    # 서버 시작 시 오류가 나면 프로세스를 멈추는 것이 좋습니다.
    # 여기서는 간단히 프린트만 합니다.

# --- 2. 이미지 폴더 경로 설정 ---
IMAGE_FOLDER = 'img'

# --- 3. API 엔드포인트(URL 경로) 정의 ---

# API 1: img 폴더 안의 이미지 파일 목록을 반환
@app.route('/get-images', methods=['GET'])
def get_images():
    try:
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        return jsonify(image_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API 2: 이미지 생성을 요청받고, 결과를 반환
@app.route('/generate-image', methods=['POST'])
def generate_image_api():
    data = request.json
    if 'filename' not in data:
        return jsonify({"error": "No filename provided"}), 400

    filename = data['filename']
    texture_path = os.path.join(IMAGE_FOLDER, filename)

    if not os.path.exists(texture_path):
        return jsonify({"error": "File not found"}), 404

    try:
        texture_image = Image.open(texture_path)

        prompt = (
            "You are a brilliant architect and an image generation AI."
            "Generate a photorealistic, high-quality image of a modern country house."
            "Use the image provided as the texture for the 'exterior wall'"
            "The camera composition should highlight the details of the cladding, but without being an extreme close-up."
        )

        # prompt 참고
            # It is crucial that the image also captures the building's overall design and feel.

        # Gemini API 호출
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=[prompt, texture_image],
            config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
            )
        )

        # 생성된 이미지를 Base64로 인코딩하여 전달
        # 응답 구조는 모델 버전에 따라 다를 수 있습니다.
        
        generated_image_pil = None  # 생성된 이미지를 담을 변수 초기화
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                # 찾은 이미지를 변수에 저장하고 반복문을 빠져나옵니다.
                generated_image_pil = Image.open(BytesIO(part.inline_data.data))
                break # 이미지를 찾았으니 더 이상 반복할 필요가 없습니다.

        # 이미지가 정상적으로 생성되었는지 확인
        if generated_image_pil:
            # 이미지를 Base64 문자열로 변환 (웹 전송을 위해)
            buffered = BytesIO()
            generated_image_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # JSON 형식으로 최종 응답 반환
            return jsonify({"image": "data:image/png;base64," + img_str})
        else:
            # AI 응답에 이미지가 없는 경우에 대한 예외 처리
            return jsonify({"error": "AI response did not contain an image"}), 500

    except Exception as e:
        print(f"Error during image generation: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

# API 3: img 폴더의 정적 파일을 서비스하기 위함 (미리보기 이미지용)
@app.route('/img/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


# --- 4. 서버 실행 ---
if __name__ == '__main__':
    # 디버그 모드로 실행, 실제 배포 시에는 debug=False로 변경
    app.run(debug=True, port=5000)