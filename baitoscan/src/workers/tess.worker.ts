import { recognize, type RecognizeResult } from "tesseract.js";

type WorkerMessage =
  | { type: "load"; payload?: { lang?: string } }
  | { type: "recognize"; id: string; payload: Blob | File };

interface ReadyMessage {
  type: "ready";
}

interface ProgressMessage {
  type: "progress";
  id: string | null;
  progress: number;
  status: string;
}

interface ResultMessage {
  type: "result";
  id: string;
  text: string;
  confidence: number;
  raw: RecognizeResult["data"];
}

interface ErrorMessage {
  type: "error";
  id?: string;
  message: string;
}

type OutgoingMessage = ReadyMessage | ProgressMessage | ResultMessage | ErrorMessage;

declare const self: DedicatedWorkerGlobalScope;

let currentLang = "eng";

const post = (message: OutgoingMessage) => {
  self.postMessage(message);
};

const sendProgress = (id: string | null, progress: number, status: string) => {
  post({ type: "progress", id, progress, status });
};

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type } = event.data;

  switch (type) {
    case "load": {
      currentLang = event.data.payload?.lang ?? "eng";
      post({ type: "ready" });
      break;
    }

    case "recognize": {
      const { id, payload } = event.data;
      if (!payload) {
        post({ type: "error", id, message: "Missing image payload" });
        break;
      }

      try {
        sendProgress(id, 0, "Recognizing");
        const whitelist = "0123456789/:~";
        const result = await recognize(payload, currentLang, {
          config: {
            tessedit_char_whitelist: whitelist,
            tessedit_char_blacklist: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[]{}()<>@#$%^&*-_=+\\|;\"',.?¡¿§`~¡™£¢∞§¶•ªº–≠œ∑´®†¥¨ˆøπ“‘«åß∂ƒ©˙∆˚¬…æ≈ç√∫˜µ≤≥÷",
            classify_bln_numeric_mode: "1",
            preserve_interword_spaces: "1",
            user_defined_dpi: "300",
          },
          tessedit_char_whitelist: whitelist,
          psm: 6,
          logger: (m) => {
            const progress = typeof m.progress === "number" ? m.progress : 0;
            sendProgress(id, progress, m.status ?? "");
          },
        });
        sendProgress(id, 1, "Done");
        const lines = result.data.lines
          ?.map((line) => line.text?.trim() ?? "")
          .filter(Boolean);
        const combinedText = lines && lines.length > 0 ? lines.join("\n") : result.data.text;
        const sanitizedText = combinedText
          .replace(/[bB]/g, "6")
          .replace(/[oO]/g, "0")
          .replace(/[sS]/g, "5")
          .replace(/[lI]/g, "1")
          .replace(/[A-Za-z]/g, "");
        post({
          type: "result",
          id,
          text: sanitizedText,
          confidence: result.data.confidence ?? 0,
          raw: result.data,
        });
      } catch (err) {
        post({
          type: "error",
          id,
          message: err instanceof Error ? err.message : "Unknown error",
        });
      }
      break;
    }
  }
};

export {};
