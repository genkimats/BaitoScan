import { useMemo, useState } from "react";
import PageWrapper from "../components/Layout/PageWrapper";
import ImageCapture from "../components/Capture/ImageCapture";
import { useTesseractWorker } from "../hooks/useTesseractWorker";
import { parseWorkLine, type WorkRow } from "../lib/parse";
import { computeMonthly, type PaySettings } from "../lib/salary";

const SHIFT_PATTERN = /\d{1,2}\/\d{1,2} \d{1,2}:\d{2}\s*~\s*\d{1,2}:\d{2}/g;

const DEFAULT_SETTINGS: PaySettings = {
  weekdayRate: 1200,
  weekendRate: 1500,
  deepNightMultiplier: 1.25,
  deepNightStart: "22:00",
  deepNightEnd: "05:00",
  transitFee: 250,
};

function normalizeRecognizedText(text: string): string {
  return text
    .replace(/[／|]/g, "/")
    .replace(/[;；﹔]/g, ":")
    .replace(/(?<=\d)[．·.](?=\d)/g, ":")
    .replace(/[~﹘﹣–—]/g, "~")
    .replace(/\s*~\s*/g, " ~ ")
    .replace(/[ ]{2,}/g, " ")
    .replace(/\r?\n[ ]+/g, "\n")
    .trim();
}

export default function HomePage() {
  const { isReady, progress, status, error, recognize, resetError, previewUrl } = useTesseractWorker();
  const [recognizedText, setRecognizedText] = useState("");
  const [rows, setRows] = useState<WorkRow[]>([]);
  const [lastConfidence, setLastConfidence] = useState<number | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [parseWarnings, setParseWarnings] = useState<string[]>([]);

  const summary = useMemo(() => {
    if (rows.length === 0) return null;
    return computeMonthly(rows, DEFAULT_SETTINGS);
  }, [rows]);

  const progressPercent = Math.round(progress * 100);
  const showProgressPercent = progressPercent > 0 && progressPercent < 100;

  const handleImageSelect = async (file: File) => {
    if (!isReady) {
      alert("OCR model is still loading. Try again shortly.");
      return;
    }

    setIsProcessing(true);
    setParseWarnings([]);
    resetError();

    try {
      const { text, confidence } = await recognize(file);
      const normalizedText = normalizeRecognizedText(text);
      setRecognizedText(normalizedText);
      setLastConfidence(confidence);

      const accepted: WorkRow[] = [];
      const warnings: string[] = [];

      const matchedSegments = Array.from(normalizedText.matchAll(SHIFT_PATTERN)).map((m) => m[0]);
      const candidateLines =
        matchedSegments.length > 0
          ? matchedSegments
          : normalizedText
              .split(/\r?\n/)
              .map((line) => line.trim())
              .filter(Boolean);

      for (const line of candidateLines) {
        const parsed = parseWorkLine(line);
        if (parsed) {
          accepted.push(parsed);
        } else {
          warnings.push(line);
        }
      }

      if (accepted.length > 0) {
        setRows((prev) => {
          const dedup = new Map<string, WorkRow>();
          for (const row of [...prev, ...accepted]) {
            const key = `${row.date}_${row.start}_${row.end}`;
            dedup.set(key, row);
          }
          return Array.from(dedup.values()).sort((a, b) => {
            const dateDiff = a.date.localeCompare(b.date);
            return dateDiff !== 0 ? dateDiff : a.start.localeCompare(b.start);
          });
        });
      }

      if (warnings.length > 0) setParseWarnings(warnings);
    } catch (err) {
      console.error(err);
      setRecognizedText("");
      setLastConfidence(null);
      setParseWarnings(["Failed to run OCR. See console for details."]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <PageWrapper>
      <div className="grid gap-4 max-w-md mx-auto">
        <h2 className="text-xl font-semibold text-center">BaitoScan</h2>
        <ImageCapture onImageSelect={handleImageSelect} />
        <div className="text-sm text-center">
          {isReady ? (
            <span className="text-green-600">✅ OCR worker ready</span>
          ) : (
            <span className="text-gray-500">Loading OCR model…</span>
          )}
          {error && <p className="text-red-600 mt-1">Worker error: {error}</p>}
        </div>
        {isProcessing && (
          <p className="text-sm text-blue-600 text-center">
            Reading shift slip…
            {showProgressPercent ? ` (${progressPercent}%)` : ""}
            {status ? ` • ${status}` : ""}
          </p>
        )}

        {previewUrl && (
          <div className="bg-white border border-gray-200 rounded-md p-3 text-sm">
            <p className="text-xs text-gray-600 mb-2">Preprocessed image sent to Tesseract</p>
            <img src={previewUrl} alt="Preprocessed for OCR" className="w-full rounded-sm border border-gray-300" />
          </div>
        )}

        {recognizedText && (
          <div className="bg-gray-100 rounded-md p-3 text-sm">
            <p className="font-mono whitespace-pre-wrap break-words">{recognizedText}</p>
            {lastConfidence !== null && (
              <p className="text-xs text-gray-600 mt-1">confidence: {lastConfidence.toFixed(1)}%</p>
            )}
          </div>
        )}

        {parseWarnings.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-md p-3 text-sm">
            <p className="font-semibold mb-1">Couldn&apos;t parse these lines:</p>
            <ul className="list-disc list-inside space-y-1">
              {parseWarnings.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </div>
        )}

        {rows.length > 0 && summary && (
          <div className="text-left mt-2 bg-white p-3 rounded-lg shadow">
            <h3 className="font-semibold mb-2">Recognized Shifts</h3>
            <div className="space-y-1 text-sm">
              {rows.map((r) => (
                <div key={`${r.date}-${r.start}-${r.end}`}>
                  {r.date}: {r.start} ~ {r.end}
                </div>
              ))}
            </div>

            <div className="mt-3 text-sm border-t border-gray-200 pt-2 space-y-1">
              <p>Regular hours: {summary.regularHours} h</p>
              <p>Deep-night hours: {summary.nightHours} h</p>
              <p>Transit total: ¥{summary.transitTotal}</p>
              <p className="font-semibold text-lg">Estimated pay: ¥{summary.totalPay}</p>
            </div>
          </div>
        )}
      </div>
    </PageWrapper>
  );
}
