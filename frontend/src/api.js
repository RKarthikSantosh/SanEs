import axios from "axios";

export const axiosInstance = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 60000,
});

// POST /upload - starts background processing, returns task_id
export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await axiosInstance.post("/upload", form);
  return data;
}

// GET /upload/status/:taskId - poll processing status
export async function getUploadStatus(taskId) {
  const { data } = await axiosInstance.get(`/upload/status/${taskId}`);
  return data;
}

// POST /ask - hybrid retrieval + grounded answer
export async function askQuestion(question, top_n = 'auto') {
  const payload = { question };
  if (top_n && top_n !== 'auto') {
    payload.top_n = Number(top_n);
  }
  const { data } = await axiosInstance.post("/ask", payload);
  return data;
}

// GET /health - backend liveness
export async function checkHealth() {
  const { data } = await axiosInstance.get("/health");
  return data?.status === "healthy";
}
