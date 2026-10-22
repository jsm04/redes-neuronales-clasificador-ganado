import { breedLabel, type Prediction } from '@/lib/api';

type Props = {
  predictions: Prediction[];
  model: string;
};

export function PredictionResults({ predictions, model }: Props) {
  const [best, ...rest] = predictions;

  return (
    <section className="results" aria-live="polite">
      <p className="eyebrow">Resultado</p>
      <h2 className="results__breed">{breedLabel(best.breed)}</h2>
      <p className="results__confidence">
        {(best.confidence * 100).toFixed(1)}% de confianza
      </p>

      <div
        className="results__bar"
        role="img"
        aria-label={`Confianza ${(best.confidence * 100).toFixed(1)}%`}
      >
        <span style={{ width: `${best.confidence * 100}%` }} />
      </div>

      {rest.length > 0 && (
        <>
          <h3 className="results__subtitle">Otras probabilidades</h3>
          <ul className="results__list">
            {rest.map((item) => (
              <li key={item.breed}>
                <span>{breedLabel(item.breed)}</span>
                <span className="muted">
                  {(item.confidence * 100).toFixed(1)}%
                </span>
              </li>
            ))}
          </ul>
        </>
      )}

      <p className="results__meta">Modelo: {model}</p>
    </section>
  );
}
