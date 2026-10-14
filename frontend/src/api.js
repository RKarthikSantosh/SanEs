import axios from "axios";

export const axiosInstance = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 60000,
});

// POST /upload - multipart form with the file
export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await axiosInstance.post("/upload", form);
  return data;
}

// POST /ask - hybrid retrieval + grounded answer
export async function askQuestion(question, top_n = 5) {
  const { data } = await axiosInstance.post("/ask", { question, top_n });
  return data;
}

// GET /health - backend liveness
export async function checkHealth() {
  const { data } = await axiosInstance.get("/health");
  return data?.status === "healthy";
}
