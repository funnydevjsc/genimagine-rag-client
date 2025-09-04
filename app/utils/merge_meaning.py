import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import sent_tokenize  # Thư viện NLP tiếng Việt
import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import sent_tokenize  # Thư viện NLP tiếng Việt


class SemanticChunker:
    def __init__(self, min_sentences=3, max_sentences=5, similarity_threshold=0.3):
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            strip_accents=None
        )

    def split_into_sentences(self, text: str) -> List[str]:
        """Tách văn bản thành câu sử dụng underthesea"""
        text = re.sub(r'[^\w\s.;:?,(){}%\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def calculate_sentence_similarities(self, sentences: List[str]) -> np.ndarray:
        """Tính toán ma trận độ tương đồng giữa các câu"""
        sentence_vectors = self.vectorizer.fit_transform(sentences)# Vectorize các câu
        similarity_matrix = cosine_similarity(sentence_vectors)# Tính similarity matrix
        return similarity_matrix

    def find_semantic_boundaries(self, similarity_matrix: np.ndarray) -> List[int]:
        """Tìm ranh giới ngữ nghĩa dựa trên độ tương đồng"""
        n_sentences = len(similarity_matrix)
        boundaries = []
        current_start = 0

        for i in range(1, n_sentences):
            # Tính độ tương đồng trung bình với các câu trước đó trong chunk hiện tại
            avg_similarity = np.mean(similarity_matrix[current_start:i, i])
            
            # Điều kiện để tạo boundary mới:
            # 1. Độ tương đồng thấp hơn ngưỡng
            # 2. Đủ số câu tối thiểu
            # 3. Chưa vượt quá số câu tối đa
            if (avg_similarity < self.similarity_threshold and 
                i - current_start >= self.min_sentences and 
                i - current_start <= self.max_sentences):
                boundaries.append(i)
                current_start = i

        # Xử lý phần còn lại
        if current_start < n_sentences:
            boundaries.append(n_sentences)

        return boundaries

    def merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Gộp các chunk nhỏ với chunk lân cận có độ tương đồng cao nhất"""
        if len(chunks) <= 1:
            return chunks

        while True:
            # Tìm chunk nhỏ nhất
            chunk_sizes = [len(self.split_into_sentences(chunk)) for chunk in chunks]
            min_size_idx = np.argmin(chunk_sizes)
            
            if chunk_sizes[min_size_idx] >= self.min_sentences:
                break

            # Tính độ tương đồng với các chunk lân cận
            chunk_vectors = self.vectorizer.fit_transform(chunks)
            similarities = cosine_similarity(chunk_vectors)
            
            # Tìm chunk lân cận có độ tương đồng cao nhất
            neighbor_similarities = []
            if min_size_idx > 0:
                neighbor_similarities.append((min_size_idx-1, similarities[min_size_idx][min_size_idx-1]))
            if min_size_idx < len(chunks)-1:
                neighbor_similarities.append((min_size_idx+1, similarities[min_size_idx][min_size_idx+1]))
            
            if not neighbor_similarities:
                break
                
            best_neighbor_idx = max(neighbor_similarities, key=lambda x: x[1])[0]
            
            # Gộp chunks
            new_chunks = []
            for i in range(len(chunks)):
                if i == min(min_size_idx, best_neighbor_idx):
                    new_chunks.append(f"{chunks[min_size_idx]} {chunks[best_neighbor_idx]}")
                elif i != max(min_size_idx, best_neighbor_idx):
                    new_chunks.append(chunks[i])
            chunks = new_chunks

        return chunks

    def create_semantic_chunks(self, text: str) -> List[str]:
        """Tạo các chunk dựa trên ngữ nghĩa"""
        sentences = self.split_into_sentences(text) # Tách câu
        if len(sentences) <= self.min_sentences:
            return [text]

        # Tính ma trận độ tương đồng
        similarity_matrix = self.calculate_sentence_similarities(sentences)
        
        # Tìm ranh giới ngữ nghĩa
        boundaries = self.find_semantic_boundaries(similarity_matrix)
        
        # Tạo chunks từ boundaries
        chunks = []
        start = 0
        for boundary in boundaries:
            chunk = ' '.join(sentences[start:boundary])
            chunks.append(chunk)
            start = boundary
            
        # Gộp các chunk nhỏ
        chunks = self.merge_small_chunks(chunks)
        
        return chunks

    def analyze_chunk_coherence(self, chunk: str) -> float:
        """Phân tích độ liên kết của một chunk"""
        sentences = self.split_into_sentences(chunk)
        if len(sentences) <= 1:
            return 1.0
            
        vectors = self.vectorizer.fit_transform(sentences)
        similarities = cosine_similarity(vectors)
        coherence = np.mean([similarities[i][i+1] for i in range(len(sentences)-1)])# Tính độ liên kết trung bình giữa các câu liên tiếp
        return coherence

    def get_chunk_info(self, chunks: List[str]) -> None:
        """In thông tin chi tiết về các chunk"""
        for i, chunk in enumerate(chunks):
            sentences = self.split_into_sentences(chunk)
            coherence = self.analyze_chunk_coherence(chunk)
            print(f"\nChunk {i+1}:")
            print(f"Số câu: {len(sentences)}")
            print(f"Độ liên kết: {coherence:.3f}")
            print(f"Nội dung: {chunk}...")

# def main():
#     sample_text = """
#     Trí tuệ nhân tạo đang phát triển nhanh chóng trong những năm gần đây. Các ứng dụng AI ngày càng đa dạng và phổ biến trong cuộc sống. Từ chatbot cho đến xe tự lái, AI đang thay đổi cách chúng ta làm việc và sinh hoạt.

#     Tuy nhiên, sự phát triển của AI cũng đặt ra nhiều thách thức. Vấn đề đạo đức và quyền riêng tư cần được quan tâm đặc biệt. Nhiều chuyên gia lo ngại về việc lạm dụng AI vào mục đích xấu.

#     Giáo dục về AI ngày càng trở nên quan trọng. Các trường học đang đưa kiến thức về AI vào chương trình giảng dạy. Sinh viên cần được trang bị kỹ năng mới để thích nghi với thời đại số.

#     An toàn và bảo mật là những ưu tiên hàng đầu trong phát triển AI. Các công ty công nghệ đang đầu tư nhiều nguồn lực để đảm bảo AI hoạt động an toàn. Cộng đồng quốc tế cũng đang xây dựng các tiêu chuẩn và quy định về AI.
#     """

#     chunker = SemanticChunker(
#         min_sentences=2,
#         max_sentences=5,
#         similarity_threshold=0.3
#     )
    
#     chunks = chunker.create_semantic_chunks(sample_text)
#     chunker.get_chunk_info(chunks)

# if __name__ == "__main__":
#     main()