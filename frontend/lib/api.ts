const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export type Prediction = {
  breed: string;
  confidence: number;
};

export type PredictionResponse = {
  model: string;
  predictions: Prediction[];
};

// Nombres legibles para las carpetas de clase del checkpoint.
export const BREED_LABELS: Record<string, string> = {
  'ayrshire-cattle': 'Ayrshire',
  'brown-swiss-cattle': 'Brown Swiss',
  'holstein-friesian-cattle': 'Holstein Friesian',
  'jersey-cattle': 'Jersey',
  'red-dane-cattle': 'Red Dane',
};

export function breedLabel(breed: string): string {
  return BREED_LABELS[breed] ?? breed;
}

export async function predictBreed(
  file: File,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const body = new FormData();
  body.append('file', file);

  const response = await fetch(`${API_URL}/api/v1/predictions`, {
    method: 'POST',
    body,
    signal,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `El servidor respondió ${response.status}.`);
  }

  return (await response.json()) as PredictionResponse;
}
