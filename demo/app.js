const API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const CHAT_MODEL = "gemini-2.5-flash";
const EMBEDDING_MODEL = "gemini-embedding-001";
const TOP_K = 3;
const MAX_FUNCTION_CALL_ROUNDS = 5;

const SYSTEM_INSTRUCTION = `You are a healthcare information assistant. You are NOT a doctor and you do not \
provide medical diagnoses. Your role is to give general, educational guidance about \
common symptoms based only on the reference material provided in each user turn.

Rules:
- Ground your answer in the "Reference" entries given in the user message context. \
Do not invent conditions or advice that aren't supported by the references or general \
safe self-care knowledge.
- If the retrieved references mark a matching condition as urgent, or the user describes \
severe/emergency symptoms (e.g. chest pain, difficulty breathing, severe bleeding, loss of \
consciousness), clearly advise the user to seek immediate professional or emergency care, \
and call the \`schedule_appointment\` tool.
- After giving the user a suggested condition and advice, call the \`log_symptom_check\` tool \
to record the interaction.
- You may call \`get_patient_history\` if knowing the user's recent symptom-check history would \
help you give better guidance.
- Always end your response with a short disclaimer that this is educational information, not \
a medical diagnosis, and that the user should consult a qualified healthcare professional for \
real medical concerns.`;

const FUNCTION_DECLARATIONS = [
  {
    name: "log_symptom_check",
    description:
      "Record a symptom-check interaction for the current user: the symptoms reported " +
      "and the condition/advice suggested. Call this after giving the user a suggested " +
      "condition and advice.",
    parametersJsonSchema: {
      type: "object",
      properties: {
        symptoms: { type: "string", description: "Comma-separated symptoms, e.g. 'fever, sore throat'." },
        predicted_condition: { type: "string", description: "Likely condition suggested, e.g. 'Common cold'." },
        advice_given: { type: "string", description: "Advice given to the user." },
      },
      required: ["symptoms", "predicted_condition", "advice_given"],
    },
  },
  {
    name: "get_patient_history",
    description: "Retrieve the current user's most recent symptom-check history, if any.",
    parametersJsonSchema: {
      type: "object",
      properties: {
        limit: { type: "integer", minimum: 1, maximum: 20, description: "Maximum number of past entries to return." },
      },
      required: [],
    },
  },
  {
    name: "schedule_appointment",
    description:
      "Request a follow-up appointment with a human clinician for the current user. " +
      "Use this for symptoms that are urgent or beyond general self-care advice.",
    parametersJsonSchema: {
      type: "object",
      properties: {
        reason: { type: "string", description: "Reason for the appointment request." },
        preferred_date: { type: "string", description: "Preferred date in YYYY-MM-DD format, or 'unspecified'." },
      },
      required: ["reason", "preferred_date"],
    },
  },
];

let apiKey = null;
let kb = null; // { embedding_model, entries: [{id, symptoms, condition, advice, urgent, embedding}] }

const keyPanel = document.getElementById("key-panel");
const keyForm = document.getElementById("key-form");
const keyInput = document.getElementById("key-input");
const keyError = document.getElementById("key-error");
const chatEl = document.getElementById("chat");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");

function addBubble(text, role) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  chatEl.appendChild(bubble);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

function docText(entry) {
  return `Symptoms: ${entry.symptoms.join(", ")}. Likely condition: ${entry.condition}. Advice: ${entry.advice}`;
}

async function embedQuery(text) {
  const res = await fetch(`${API_BASE}/${EMBEDDING_MODEL}:embedContent`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
    body: JSON.stringify({ content: { parts: [{ text }] }, taskType: "RETRIEVAL_QUERY" }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Embedding request failed (${res.status}): ${body.slice(0, 200)}`);
  }
  const data = await res.json();
  return data.embedding.values;
}

function retrieveTopK(queryVec) {
  const scored = kb.entries.map((entry) => ({
    entry,
    score: cosineSimilarity(queryVec, entry.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, TOP_K).map((s) => s.entry);
}

function formatContext(entries) {
  return entries
    .map(
      (entry, i) =>
        `Reference ${i + 1}: symptoms=[${entry.symptoms.join(", ")}] condition="${entry.condition}" ` +
        `advice="${entry.advice}" urgent=${entry.urgent}`
    )
    .join("\n");
}

function getUserId() {
  let userId = localStorage.getItem("healthcare_assistant_user_id");
  if (!userId) {
    userId = crypto.randomUUID();
    localStorage.setItem("healthcare_assistant_user_id", userId);
  }
  return userId;
}

function loadTable(name) {
  return JSON.parse(localStorage.getItem(name) || "[]");
}

function saveTable(name, rows) {
  localStorage.setItem(name, JSON.stringify(rows));
}

function toolLogSymptomCheck(userId, args) {
  const rows = loadTable("symptom_logs");
  const row = {
    id: rows.length + 1,
    user_id: userId,
    timestamp: new Date().toISOString(),
    symptoms: args.symptoms,
    predicted_condition: args.predicted_condition,
    advice_given: args.advice_given,
  };
  rows.push(row);
  saveTable("symptom_logs", rows);
  return { status: "ok", log_id: row.id };
}

function toolGetPatientHistory(userId, args) {
  const limit = args.limit || 5;
  const rows = loadTable("symptom_logs")
    .filter((r) => r.user_id === userId)
    .slice(-limit)
    .reverse();
  return { status: "ok", history: rows };
}

function toolScheduleAppointment(userId, args) {
  const rows = loadTable("appointments");
  const row = {
    id: rows.length + 1,
    user_id: userId,
    requested_at: new Date().toISOString(),
    reason: args.reason,
    preferred_date: args.preferred_date,
    status: "pending",
  };
  rows.push(row);
  saveTable("appointments", rows);
  return { status: "ok", appointment_id: row.id };
}

function dispatchTool(name, args, userId) {
  switch (name) {
    case "log_symptom_check":
      return toolLogSymptomCheck(userId, args);
    case "get_patient_history":
      return toolGetPatientHistory(userId, args);
    case "schedule_appointment":
      return toolScheduleAppointment(userId, args);
    default:
      return { error: `unknown tool ${name}` };
  }
}

async function callGemini(contents) {
  const res = await fetch(`${API_BASE}/${CHAT_MODEL}:generateContent`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM_INSTRUCTION }] },
      contents,
      tools: [{ functionDeclarations: FUNCTION_DECLARATIONS }],
    }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Gemini request failed (${res.status}): ${body.slice(0, 200)}`);
  }
  return res.json();
}

async function ask(query) {
  const userId = getUserId();
  const queryVec = await embedQuery(query);
  const references = retrieveTopK(queryVec);
  const contextBlock = formatContext(references);

  const contents = [
    { role: "user", parts: [{ text: `${contextBlock}\n\nUser question: ${query}` }] },
  ];

  const collectedText = [];

  for (let round = 0; round < MAX_FUNCTION_CALL_ROUNDS; round++) {
    const data = await callGemini(contents);
    const modelContent = data.candidates[0].content;
    const parts = modelContent.parts || [];

    const textParts = parts.filter((p) => p.text).map((p) => p.text);
    if (textParts.length) collectedText.push(textParts.join(""));

    const functionCalls = parts.filter((p) => p.functionCall).map((p) => p.functionCall);
    if (functionCalls.length === 0) {
      return collectedText.join("\n\n");
    }

    contents.push(modelContent);
    const responseParts = functionCalls.map((call) => ({
      functionResponse: { name: call.name, response: dispatchTool(call.name, call.args || {}, userId) },
    }));
    contents.push({ role: "user", parts: responseParts });
  }

  collectedText.push("I couldn't finish that after several tool calls — please rephrase or consult a clinician.");
  return collectedText.join("\n\n");
}

keyForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = keyInput.value.trim();
  if (!value) return;
  apiKey = value;
  keyError.textContent = "";

  try {
    kb = await (await fetch("kb_embeddings.json")).json();
  } catch (err) {
    keyError.textContent = `Failed to load knowledge base: ${err.message}`;
    apiKey = null;
    return;
  }

  keyPanel.classList.add("hidden");
  chatEl.classList.remove("hidden");
  chatForm.classList.remove("hidden");
  messageInput.focus();
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  addBubble(message, "user");
  messageInput.value = "";
  messageInput.disabled = true;
  const pending = addBubble("Thinking...", "assistant pending");

  try {
    const reply = await ask(message);
    pending.textContent = reply;
    pending.classList.remove("pending");
  } catch (err) {
    pending.textContent = `Error: ${err.message}`;
    pending.classList.remove("pending", "assistant");
    pending.classList.add("error");
  } finally {
    messageInput.disabled = false;
    messageInput.focus();
  }
});
