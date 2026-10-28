'use client';

import { useEffect, useState } from 'react';

import { ImageUpload } from '@/components/image-upload';
import { PredictionResults } from '@/components/prediction-results';
import { predictBreed, type PredictionResponse } from '@/lib/api';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function select(next: File) {
    setFile(next);
    setResult(null);
    setError(null);
  }

  async function submit() {
    if (!file || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      setResult(await predictBreed(file));
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'No se pudo contactar con el servicio de predicción.',
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <header className="masthead">
        <p className="eyebrow">Clasificador de imagenes</p>
        <h1>Identificación de razas bovinas</h1>
        <p className="lede">
          Sube una fotografía y la red estimará la raza del animal con su
          probabilidad.
        </p>
      </header>

      <div className="grid">
        <section className="card">
          <ImageUpload
            previewUrl={previewUrl}
            disabled={loading}
            onSelect={select}
            onError={setError}
          />

          <button
            type="button"
            className="button button--primary"
            disabled={!file || loading}
            onClick={submit}
          >
            {loading ? 'Analizando…' : 'Predecir raza'}
          </button>

          {error && (
            <p className="alert" role="alert">
              {error}
            </p>
          )}
        </section>

        <div className="card card--result">
          {result ? (
            <PredictionResults
              predictions={result.predictions}
              model={result.model}
            />
          ) : (
            <div className="placeholder">
              <p className="eyebrow">Sin resultados</p>
              <p className="muted">
                Los resultados aparecerán aquí después de analizar la imagen.
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
