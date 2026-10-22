'use client';

import { useRef, useState } from 'react';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
const MAX_BYTES = 10 * 1024 * 1024;

type Props = {
  previewUrl: string | null;
  disabled?: boolean;
  onSelect: (file: File) => void;
  onError: (message: string) => void;
};

export function ImageUpload({ previewUrl, disabled, onSelect, onError }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function accept(file: File | undefined) {
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      onError('Formato no admitido. Usa JPG, PNG, WEBP o BMP.');
      return;
    }
    if (file.size > MAX_BYTES) {
      onError('La imagen supera los 10 MB.');
      return;
    }
    onSelect(file);
  }

  return (
    <div
      className={`dropzone${dragging ? ' dropzone--active' : ''}`}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        if (!disabled) accept(event.dataTransfer.files[0]);
      }}
    >
      {previewUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img className="preview" src={previewUrl} alt="Imagen seleccionada" />
      ) : (
        <div className="dropzone__empty">
          <span className="dropzone__icon" aria-hidden="true">
            ↑
          </span>
          <p>Arrastra una foto del animal aquí</p>
          <p className="muted">JPG, PNG, WEBP o BMP · máximo 10 MB</p>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(',')}
        className="sr-only"
        disabled={disabled}
        onChange={(event) => {
          accept(event.target.files?.[0]);
          event.target.value = '';
        }}
      />

      <button
        type="button"
        className="button button--ghost"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
      >
        {previewUrl ? 'Elegir otra imagen' : 'Seleccionar imagen'}
      </button>
    </div>
  );
}
