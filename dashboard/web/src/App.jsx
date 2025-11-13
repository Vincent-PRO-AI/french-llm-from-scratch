import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Plot from './components/Plot.jsx';
import './styles.css';

const METADATA_REFRESH_MS = 6000;
const LOSS_REFRESH_MS = 4000;
const VISUAL_REFRESH_MS = 5000;

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

const INT_FIELDS = new Set([
  'max_steps',
  'batch_size',
  'num_layers',
  'num_heads',
  'ff_hidden_dim',
  'layernorm_dim',
  'head_dim',
  'block_size',
  'vocab_size',
  'eval_interval',
  'eval_batches',
  'sample_interval',
  'sample_max_new_tokens',
  'sample_top_k',
  'checkpoint_interval',
]);

const INT_FIELDS_ALLOW_ZERO = new Set(['sample_interval', 'sample_max_new_tokens', 'sample_top_k']);

const FLOAT_FIELDS = new Set([
  'dropout',
  'lr',
  'weight_decay',
  'sample_temperature',
  'metrics_log_fraction',
]);

const TOKENIZER_INT_FIELDS = new Set(['vocab_size', 'min_frequency', 'limit_files']);
const TOKENIZER_FLOAT_FIELDS = new Set(['limit_mb']);

const MANUAL_LOCK = 'manual-lock';
const PRESET_ORDER = ['very-small', 'small', 'medium'];
const formatPresetLabel = (key) =>
  key
    .split('-')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');

const DEFAULT_TRAINING_CONFIG = {
  run_name: 'french_medium_60k',
  max_steps: '60000',
  batch_size: '4',
  block_size: '1024',
  embed_dim: '1024',
  vocab_size: '32000',
  layernorm_dim: '',
  head_dim: '',
  num_heads: '16',
  num_layers: '18',
  ff_hidden_dim: '4096',
  dropout: '0.1',
  lr: '0.0001',
  weight_decay: '0.0',
  eval_interval: '500',
  eval_batches: '16',
  metrics_log_fraction: '0.01',
  sample_interval: '1000',
  sample_max_new_tokens: '120',
  sample_temperature: '0.85',
  sample_top_k: '50',
  device: 'cuda',
  arch_preset: 'medium',
  extra_data_dirs: [],
  checkpoint_interval: '5000',
  tokenizer_path: 'trained_models/tokenizers/fineweb-32k/tokenizer.json',
  pretokenized_path: 'data_clean/mixed_tokenized.pt',
};

const DATA_DIR_PATHS = {
  test: 'data_clean/test',
  subtitles: 'data_clean/youtube',
  linkedin: 'data_clean/linkedin',
  instagram: 'data_clean/instagram',
  wikipedia: 'data_clean/wikipedia',
  fineweb: 'data_clean/fineweb_fr',
  fineweb_full: 'data_clean/fineweb',
  bloom_lm: 'data_clean/bloom_lm',
  oscar: 'data_clean/oscar_fr',
  french_pd_books: 'data_clean/french_pd_books',
};

const DEFAULT_TOKENIZER_CONFIG = {
  name: 'wiki-bpe',
  input_dir: 'data_clean',
  output_dir: 'trained_models/tokenizers',
  vocab_size: '32000',
  min_frequency: '2',
  limit_files: '',
  limit_mb: '',
  lowercase: false,
};

const generateSlug = (prefix) => {
  const rand = Math.random().toString(36).slice(2, 8);
  const stamp = new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 8);
  return `${prefix}-${stamp}-${rand}`;
};

function formatDate(value) {
  if (!value) return '-';
  try {
    return new Date(value).toLocaleString();
  } catch (err) {
    return value;
  }
}

function formatNumber(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} G`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)} k`;
  return value.toLocaleString();
}

function formatBytes(bytes) {
  if (!bytes || Number.isNaN(bytes)) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = -1;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value < 10 ? 2 : 1)} ${units[unitIndex]}`;
}

function StatusBadge({ status }) {
  if (!status) return <span className="small">-</span>;
  const normalized = String(status).toLowerCase();
  let cls = 'status-badge ';
  if (normalized === 'completed') cls += 'status-completed';
  else if (normalized === 'failed') cls += 'status-failed';
  else if (normalized === 'running') cls += 'status-running';
  else cls += 'status-idle';
  return <span className={cls}>{status}</span>;
}

function RunDetails({ metadata, onResume, resumeDisabled }) {
  const run = metadata?.run;
  const model = metadata?.model;
  const hasResume = Boolean(metadata?.checkpoint_path && typeof onResume === 'function');
  const resumeLabel = resumeDisabled ? 'Reprise en cours…' : "Continuer l'entraînement";
  return (
    <section>
      <h2>Détails du run</h2>
      <div className="grid two-columns">
        <div>
          <div className="label">Nom</div>
          <div className="value">{run?.run_name || '-'}</div>
        </div>
        <div>
          <div className="label">Statut</div>
          <div className="value">
            <StatusBadge status={run?.status} />
          </div>
        </div>
        <div>
          <div className="label">Début</div>
          <div className="value">{formatDate(run?.started_at)}</div>
        </div>
        <div>
          <div className="label">Fin</div>
          <div className="value">{formatDate(run?.ended_at)}</div>
        </div>
        <div>
          <div className="label">Device</div>
          <div className="value">{run?.device || model?.device || '-'}</div>
        </div>
        <div>
          <div className="label">GPU</div>
          <div className="value">{describeGpu(run?.gpu)}</div>
        </div>
      </div>
      {hasResume ? (
        <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'flex-end' }}>
          <button type="button" onClick={onResume} disabled={resumeDisabled}>
            {resumeLabel}
          </button>
        </div>
      ) : null}
    </section>
  );
}

function describeGpu(info) {
  if (!info || !info.name) return '-';
  const gb = info.total_memory_bytes
    ? `${(info.total_memory_bytes / 1024 ** 3).toFixed(1)} GB`
    : '';
  return `${info.name}${gb ? ` (${gb})` : ''}`;
}

const DATA_TASKS = [
  {
    key: 'test',
    title: '🧪 Test Corpus (Quick Start)',
    description: 'Petit corpus de test pour validation rapide du pipeline.',
    icon: '🧪',
    disabled: false,
  },
  {
    key: 'subtitles',
    title: 'Sous-titres YouTube',
    description: 'Convertit les fichiers .vtt en texte nettoyé (un .txt par vidéo).',
    icon: '📝',
  },
  {
    key: 'linkedin',
    title: 'LinkedIn (messages & posts)',
    description: 'Nettoie commentaires, partages et messages privés en texte brut.',
    icon: '💬',
  },
  {
    key: 'instagram',
    title: 'Instagram (inbox & comments)',
    description: 'Formate les discussions Inbox et les commentaires publics en texte.',
    icon: '📸',
  },
  {
    key: 'wikipedia',
    title: 'Wikipedia (IA topics)',
    description: 'Nettoie les articles importés (suppression Latex, gabarits, etc.).',
    icon: '📚',
  },
  {
    key: 'fineweb',
    title: 'FineWeb (FR) 🔄 EN COURS',
    description: 'Télécharge un extrait francophone via Hugging Face (filtrage langdetect).',
    icon: '🍷',
    buildPayload: () => {
      const defaultDocs = 2000;
      const input = window.prompt(
        'Nombre de documents français à récupérer (limiter pour éviter des téléchargements massifs) :',
        String(defaultDocs),
      );
      if (input === null) return null;
      const docs = Number.parseInt(input, 10);
      if (!Number.isFinite(docs) || docs <= 0) {
        window.alert('Valeur invalide. Téléchargement annulé.');
        return null;
      }

      const configInput = window.prompt(
        'Config FineWeb (ex: sample-10BT ou data/CC-MAIN-2024-10). Laisser vide pour le défaut sample-10BT :',
        'sample-10BT',
      );
      if (configInput === null) return null;

      const limitMiBInput = window.prompt(
        'Limite approximative en MiB (par défaut 200 MiB). Laisser vide pour conserver la valeur par défaut :',
        '',
      );
      let maxBytes;
      if (limitMiBInput && limitMiBInput.trim()) {
        const limitMiB = Number.parseFloat(limitMiBInput);
        if (Number.isFinite(limitMiB) && limitMiB > 0) {
          maxBytes = Math.floor(limitMiB * 1024 * 1024);
        } else {
          window.alert('Limite invalide, conservation de la valeur par défaut.');
        }
      }

      const payload = { max_docs: docs };
      if (configInput && configInput.trim()) {
        payload.config = configInput.trim();
      }
      if (maxBytes) {
        payload.max_bytes = maxBytes;
      }
      return payload;
    },
  },
  {
    key: 'french_pd_books',
    title: 'French-PD-Books (PleIAs)',
    description: 'Corpus de livres français libres de droit (Hugging Face).',
    icon: '📖',
    buildPayload: () => {
      const maxDocs = window.prompt(
        'Nombre max de documents (laisser vide pour tous):',
        '5000',
      );
      if (maxDocs === null) return null;
      const payload = {};
      if (maxDocs && maxDocs.trim()) {
        const parsed = Number.parseInt(maxDocs, 10);
        if (Number.isFinite(parsed) && parsed > 0) {
          payload.max_docs = parsed;
        }
      }
      return payload;
    },
  },
  {
    key: 'bloom_lm',
    title: 'Bloom Library',
    description: 'Utilisez scripts/download_bloom_lm.py pour générer les fichiers localement.',
    icon: '🌸',
    disabled: true,
  },
  {
    key: 'oscar',
    title: 'OSCAR (langue seule)',
    description: 'Télécharge un sous-corpus OSCAR (langue préfiltrée).',
    icon: '📚',
    buildPayload: () => {
      const dataset = window.prompt(
        'Nom du dataset Hugging Face (ex: oscar-corpus/OSCAR-23.01):',
        'oscar-corpus/OSCAR-23.01',
      );
      if (dataset === null || !dataset.trim()) return null;

      const language = window.prompt('Code langue (ex: fr):', 'fr');
      if (language === null || !language.trim()) return null;

      const maxDocsInput = window.prompt(
        'Limiter le nombre de documents (laisser vide pour tout télécharger):',
        '',
      );
      const payload = { dataset: dataset.trim(), language: language.trim() };
      if (maxDocsInput && maxDocsInput.trim()) {
        const parsed = Number.parseInt(maxDocsInput, 10);
        if (Number.isFinite(parsed) && parsed > 0) {
          payload.max_docs = parsed;
        }
      }
      return payload;
    },
  },
];

function progressRatio(details) {
  const total = Number(details?.total || 0);
  const processed = Number(details?.processed || 0);
  if (!total || processed > total) return null;
  return Math.min(100, Math.max(0, Math.floor((processed / total) * 100)));
}

function DataPreparationPanel({ statuses, catalog, history, onStart }) {
  catalog = catalog || {};
  history = history || {};
  return (
    <section>
      <h2>Préparation des données</h2>
      <div className="small">
        Gérez vos pipelines de preprocessing. Chaque tâche affiche sa progression en temps réel.
      </div>
      <div className="data-table">
        <div className="data-table__header">
          <div>Source</div>
          <div>Données nettoyées</div>
          <div>Statut</div>
          <div></div>
        </div>
        {DATA_TASKS.map((task) => {
          const status = statuses?.[task.key] || {};
          const jobStatus = status.status || 'idle';
          const details = status.details || {};
          const percent = progressRatio(details);
          const isDisabled = task.disabled;
          const isRunning = jobStatus === 'running';
          const catalogInfo = catalog?.[task.key] || {};
          const sizeBytes = catalogInfo.size_bytes || 0;
          const detailSize = details.size_bytes || details.sizeBytes || 0;
          const counts = catalogInfo.counts || {};
          const metaRows = [];
          const hasCatalogCounts = Object.keys(counts).length > 0;
          const historyEntry = history?.[task.key];

          if (task.key === 'subtitles') {
            if (details.last_file) metaRows.push({ label: 'Dernier fichier', value: details.last_file });
            if (details.last_output) metaRows.push({ label: 'Dernière sortie', value: details.last_output });
          } else if (task.key === 'linkedin' && !hasCatalogCounts) {
            if (typeof details.comments === 'number') metaRows.push({ label: 'Commentaires', value: details.comments });
            if (typeof details.shares === 'number') metaRows.push({ label: 'Partages', value: details.shares });
            if (typeof details.conversations === 'number') metaRows.push({ label: 'Messages', value: details.conversations });
          } else if (task.key === 'instagram' && !hasCatalogCounts) {
            if (typeof details.conversations === 'number') metaRows.push({ label: 'Conversations', value: details.conversations });
            if (typeof details.comments === 'number') metaRows.push({ label: 'Commentaires', value: details.comments });
          } else if (task.key === 'wikipedia') {
            if (!hasCatalogCounts && typeof details.processed_files === 'number') {
              metaRows.push({ label: 'Articles nettoyés', value: details.processed_files });
            }
            if (!hasCatalogCounts && typeof details.removed_lines === 'number') {
              metaRows.push({ label: 'Lignes supprimées', value: details.removed_lines });
            }
            if (typeof details.cleaned_bytes === 'number') {
              metaRows.push({ label: 'Taille nettoyée', value: formatBytes(details.cleaned_bytes) });
            }
            if (typeof details.elapsed_seconds === 'number') {
              metaRows.push({ label: 'Durée', value: `${details.elapsed_seconds} s` });
            }
          } else if (task.key === 'fineweb') {
            if (typeof details.kept_docs === 'number') {
              metaRows.push({ label: 'Documents FR', value: details.kept_docs });
            }
            if (typeof details.skipped_lang === 'number') {
              metaRows.push({ label: 'Rejets langue', value: details.skipped_lang });
            }
            if (typeof details.bytes_written === 'number') {
              metaRows.push({ label: 'Taille', value: formatBytes(details.bytes_written) });
            }
            if (details.output_path) {
              metaRows.push({ label: 'Sortie', value: details.output_path });
            }
            if (details.config) {
              metaRows.push({ label: 'Config', value: details.config });
            }
          } else if (task.key === 'oscar') {
            if (typeof details.kept_docs === 'number') {
              metaRows.push({ label: 'Documents', value: details.kept_docs });
            }
            if (typeof details.bytes_written === 'number') {
              metaRows.push({ label: 'Taille', value: formatBytes(details.bytes_written) });
            }
            if (details.dataset) {
              metaRows.push({ label: 'Dataset', value: details.dataset });
            }
            if (details.language) {
              metaRows.push({ label: 'Langue', value: details.language });
            }
            if (details.output_path) {
              metaRows.push({ label: 'Sortie', value: details.output_path });
            }
          }
          if (historyEntry?.completed_at) {
            metaRows.push({ label: 'Dernier nettoyage', value: formatDate(historyEntry.completed_at) });
          }
          if (historyEntry?.status && historyEntry.status !== 'completed') {
            metaRows.push({ label: 'Statut précédent', value: historyEntry.status });
          }
          if (!metaRows.length && (sizeBytes || detailSize)) {
            metaRows.push({ label: 'Taille', value: formatBytes(sizeBytes || detailSize) });
          }
          if (!metaRows.length && details.last_file) {
            metaRows.push({ label: 'Dernier fichier', value: details.last_file });
          }
          if (!metaRows.length && details.last_output) {
            metaRows.push({ label: 'Dernière sortie', value: details.last_output });
          }
          if (!metaRows.length) {
            metaRows.push({ label: 'Info', value: 'En attente de données' });
          }
          return (
            <div key={task.key} className="data-table__row">
              <div className="data-table__cell data-table__cell--source">
                <span className="data-icon" aria-hidden>{task.icon}</span>
                <div>
                  <div className="data-title">{task.title}</div>
                  <div className="data-desc">{task.description}</div>
                </div>
              </div>
              <div className="data-table__cell data-table__cell--output">
                <div className="data-output">
                  <div className="data-output__summary">
                    {sizeBytes ? formatBytes(sizeBytes) : 'Aucune donnée'}
                  </div>
                  <div className="data-output__details">
                    {Object.entries(counts).map(([label, value]) => {
                      const prettyLabel = label.replace(/_/g, ' ');
                      const prettyValue = typeof value === 'number' ? formatNumber(value) : value;
                      return <span key={label}>{`${prettyLabel}: ${prettyValue}`}</span>;
                    })}
                  </div>
                  {metaRows.map((row, idx) => {
                    const displayValue = typeof row.value === 'number' ? formatNumber(row.value) : row.value;
                    return (
                      <div key={idx} className="data-output__meta">
                        <span className="label">{row.label}</span>
                        <span className="value truncate">{displayValue}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="data-table__cell data-table__cell--status">
                <StatusBadge status={jobStatus} />
                <div className="data-progress">
                  {percent !== null ? (
                    <div className="progress-wrapper">
                      <div className="progress-bar">
                        <span style={{ width: `${percent}%` }} />
                      </div>
                      <span className="progress-value">{percent}%</span>
                    </div>
                  ) : (
                    <span className="value">-</span>
                  )}
                </div>
              </div>
              <div className="data-table__cell data-table__cell--action">
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => onStart(task)}
                  disabled={isDisabled || isRunning}
                >
                  {isRunning ? 'En cours…' : 'Lancer'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

const TRAINING_FIELDS = [
  {
    key: 'run_name',
    label: 'Nom du run',
    type: 'text',
    placeholder: 'Optionnel',
  },
  {
    key: 'device',
    label: 'Device',
    type: 'select',
    options: [
      { value: 'auto', label: 'Auto (CUDA si possible)' },
      { value: 'cuda', label: 'CUDA' },
      { value: 'cpu', label: 'CPU' },
    ],
    helper: 'Auto choisit CUDA si disponible côté backend.',
  },
  {
    key: 'max_steps',
    label: 'Max steps',
    type: 'number',
    min: 1,
    helper: 'Nombre total d’étapes (> 0).',
  },
  {
    key: 'batch_size',
    label: 'Batch size',
    type: 'number',
    min: 1,
    helper: 'Taille de lot par itération.',
  },
  {
    key: 'block_size',
    label: 'Block size',
    type: 'number',
    min: 32,
    helper: 'Contexte (longueur de séquence).',
  },
  {
    key: 'dropout',
    label: 'Dropout',
    type: 'number',
    step: '0.05',
    placeholder: '0.1',
  },
  {
    key: 'lr',
    label: 'Learning rate',
    type: 'number',
    step: '0.0001',
  },
  {
    key: 'weight_decay',
    label: 'Weight decay',
    type: 'number',
    step: '0.0001',
  },
  {
    key: 'eval_interval',
    label: 'Eval interval',
    type: 'number',
    min: 1,
  },
  {
    key: 'eval_batches',
    label: 'Eval batches',
    type: 'number',
    min: 1,
  },
  {
    key: 'metrics_log_fraction',
    label: 'Log fraction',
    type: 'number',
    step: '0.01',
    helper: 'Fréquence d’écriture des métriques (0-1).',
  },
  {
    key: 'sample_interval',
    label: 'Sample interval',
    type: 'number',
    min: 0,
    helper: '0 pour désactiver les échantillons.',
  },
  {
    key: 'sample_max_new_tokens',
    label: 'Sample tokens',
    type: 'number',
    min: 1,
  },
  {
    key: 'sample_temperature',
    label: 'Sample temp',
    type: 'number',
    step: '0.05',
  },
  {
    key: 'sample_top_k',
    label: 'Sample top-k',
    type: 'number',
    min: 0,
    helper: '0 = pas de top-k.',
  },
];

function TokenizerPanel({
  config,
  job,
  onConfigChange,
  onStart,
  onTest,
  testing,
  testInput,
  onTestInputChange,
  existing,
  selectedTokenizer,
  onSelectTokenizer,
  testResult,
}) {
  const status = job?.status || 'idle';
  const details = job?.details || {};
  const processed = Number(details.processed ?? details.processed_files ?? 0);
  const total = Number(details.total_files ?? details.total ?? 0);
  const percent = total > 0 ? Math.min(100, Math.floor((processed / total) * 100)) : null;
  const cleanedBytes = details.cleaned_bytes ?? details.cleanedBytes;
  const skippedFiles = details.skipped_files ?? details.skippedFiles;
  const message = job?.message;
  const isRunning = status === 'running';

  const update = (field) => (event) => onConfigChange(field, event.target.value);

  return (
    <section>
      <h2>Tokenizer BPE</h2>
      <div className="small">
        Nettoie d'abord Wikipedia, puis entraîne un tokenizer BPE pour calibrer vocabulaire et architecture.
      </div>
      <div className="tokenizer-grid">
        <div className="tokenizer-card">
          <label className="label" htmlFor="tokenizer-name">Nom du tokenizer</label>
          <input id="tokenizer-name" type="text" value={config.name ?? ''} onChange={update('name')} />
        </div>
        <div className="tokenizer-card">
          <label className="label" htmlFor="tokenizer-input">Répertoire d'entrée</label>
          <input id="tokenizer-input" type="text" value={config.input_dir ?? ''} onChange={update('input_dir')} />
          <small>Utilise toutes les sources nettoyées (ex: data_clean).</small>
        </div>
        <div className="tokenizer-card">
          <label className="label" htmlFor="tokenizer-output">Répertoire de sortie</label>
          <input id="tokenizer-output" type="text" value={config.output_dir ?? ''} onChange={update('output_dir')} />
        </div>
        <div className="tokenizer-card">
          <label className="label" htmlFor="tokenizer-vocab">Vocab size</label>
          <input id="tokenizer-vocab" type="number" min="256" step="512" value={config.vocab_size ?? ''} onChange={update('vocab_size')} />
          <small>Mini GPT ≈ 16k · Small GPT ≈ 32k.</small>
        </div>
        <div className="tokenizer-card">
          <label className="label" htmlFor="tokenizer-minfreq">Min fréquence</label>
          <input id="tokenizer-minfreq" type="number" min="1" value={config.min_frequency ?? ''} onChange={update('min_frequency')} />
        </div>
        <div className="tokenizer-card">
          <label className="label" htmlFor="tokenizer-limit-files">Limiter (fichiers)</label>
          <input id="tokenizer-limit-files" type="number" min="1" value={config.limit_files ?? ''} onChange={update('limit_files')} placeholder="Optionnel" />
        </div>
        <div className="tokenizer-card">
          <label className="label" htmlFor="tokenizer-limit-mb">Limiter taille (MiB)</label>
          <input id="tokenizer-limit-mb" type="number" min="1" step="10" value={config.limit_mb ?? ''} onChange={update('limit_mb')} placeholder="Optionnel" />
        </div>
        <div className="tokenizer-card tokenizer-checkbox">
          <label className="checkbox">
            <input type="checkbox" checked={Boolean(config.lowercase)} onChange={(event) => onConfigChange('lowercase', event.target.checked)} />
            <span>Passer en minuscules</span>
          </label>
        </div>
      </div>

      <div className="tokenizer-status-row">
        <div className="tokenizer-status-meta">
          <StatusBadge status={status} />
          {percent !== null ? (
            <div className="progress-wrapper">
              <div className="progress-bar">
                <span style={{ width: `${percent}%` }} />
              </div>
              <span className="progress-value">{percent}%</span>
            </div>
          ) : null}
          <div className="small">
            {total ? `${processed}/${total} fichiers` : 'En attente…'}
            {typeof skippedFiles === 'number' && skippedFiles > 0 ? ` · ignorés ${skippedFiles}` : ''}
            {typeof cleanedBytes === 'number' ? ` · ${formatBytes(cleanedBytes)}` : ''}
            {message ? ` · ${message}` : ''}
          </div>
        </div>
        <button type="button" onClick={onStart} disabled={isRunning}>
          {isRunning ? 'Entraînement en cours…' : 'Entraîner le tokenizer'}
        </button>
      </div>

      {existing && existing.length ? (
        <div className="tokenizer-available">
          <strong>Tokenizers disponibles:</strong>
          {existing.map((tk) => (
            <span key={tk.path} className="badge">
              {tk.name} · vocab {formatNumber(tk.vocab_size || 0)} · {formatDate(tk.trained_at)}
            </span>
          ))}
        </div>
      ) : null}

      <div className="tokenizer-test">
        <div className="tokenizer-test-header">
          <label className="label" htmlFor="tokenizer-select">Tokenizer à tester</label>
          <select id="tokenizer-select" value={selectedTokenizer || ''} onChange={(event) => onSelectTokenizer(event.target.value)}>
            <option value="">(Dernier tokenizer)</option>
            {existing.map((tk) => {
              const value = tk.tokenizer_path || tk.path;
              return (
                <option key={tk.path} value={value}>
                  {tk.name} · vocab {formatNumber(tk.vocab_size || 0)}
                </option>
              );
            })}
          </select>
        </div>
        <label className="label" htmlFor="tokenizer-test-input">Texte de test</label>
        <textarea
          id="tokenizer-test-input"
          value={testInput}
          onChange={(event) => onTestInputChange(event.target.value)}
          placeholder="Entrez une phrase pour visualiser la tokenisation..."
        />
        <div className="tokenizer-test-actions">
          <button type="button" onClick={onTest} disabled={testing || !testInput.trim()}>
            {testing ? 'Tokenisation…' : 'Tester le tokenizer'}
          </button>
        </div>
        {testResult ? (
          <div className={`tokenizer-test-result${testResult.ok ? '' : ' tokenizer-test-error'}`}>
            {testResult.ok ? (
              <div>
                <div><strong>Chemin:</strong> {testResult.data.tokenizer_path}</div>
                <div><strong>IDs:</strong> {testResult.data.ids.join(', ')}</div>
                <div><strong>Tokens bruts:</strong> {testResult.data.tokens.join(' | ')}</div>
                {Array.isArray(testResult.data.tokens_pretty) ? (
                  <div><strong>Segments originaux:</strong> {testResult.data.tokens_pretty.join(' | ')}</div>
                ) : null}
              </div>
            ) : (
              <div><strong>Erreur:</strong> {testResult.message}</div>
            )}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function ArchitectureBuilder({ config, presets = {}, onConfigChange, onApplyPreset }) {
  const toPositive = (value) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  };
  const numLayers = Number(config.num_layers || 0);
  const embedDim = Number(config.embed_dim || 0);
  const numHeads = Number(config.num_heads || 0);
  const ffDim = Number(config.ff_hidden_dim || 0);
  const layerNormDim = toPositive(config.layernorm_dim) ?? embedDim;
  const headDim = toPositive(config.head_dim) ?? layerNormDim;
  const vocabSize = Number(config.vocab_size || 256) || 256;
  const summary = buildSummaryFromConfig(config);
  const totalParams = summary?.total_params || 0;

  const presetEntries = Object.entries(presets).sort((a, b) => {
    const ia = PRESET_ORDER.indexOf(a[0]);
    const ib = PRESET_ORDER.indexOf(b[0]);
    if (ia !== -1 && ib !== -1) return ia - ib;
    if (ia !== -1) return -1;
    if (ib !== -1) return 1;
    return a[0].localeCompare(b[0]);
  });
  const activePreset = config.arch_preset || '';
  const preset = activePreset ? presets[activePreset] : undefined;
  const presetModified = preset
    ? Object.entries(preset).some(([key, value]) => {
        if (!Number.isFinite(value)) return false;
        const current = Number(config[key]);
        return Number(current) !== Number(value);
      })
    : false;
  const presetStatus = activePreset
    ? `Preset actuel : ${activePreset}${presetModified ? ' (modifié)' : ''}`
    : 'Configuration personnalisée';

  const handlePresetSelect = (event) => {
    const key = event.target.value;
    if (onApplyPreset) onApplyPreset(key);
  };

  const increment = (field, delta, minValue = 0) => {
    const current = Number(config[field] || 0) || 0;
    const next = Math.max(minValue, current + delta);
    onConfigChange(field, String(next));
  };

  return (
    <section>
      <h2>Architecture du transformer</h2>
      {presetEntries.length ? (
        <div className="architecture-preset">
          <label className="label" htmlFor="arch-preset-select">Preset d'architecture</label>
          <div className="preset-row">
            <select id="arch-preset-select" value={activePreset} onChange={handlePresetSelect}>
              <option value="">Personnalisé</option>
              {presetEntries.map(([key]) => (
                <option key={key} value={key}>
                  {formatPresetLabel(key)}
                </option>
              ))}
            </select>
            <span className="preset-status">{presetStatus}</span>
            {activePreset ? (
              <button
                type="button"
                className="ghost-button"
                onClick={() => onApplyPreset && onApplyPreset(activePreset)}
              >
                Ré-appliquer
              </button>
            ) : null}
          </div>
          <small>Applique un gabarit (dimension, couches, vocabulaire) modifiable ensuite.</small>
        </div>
      ) : null}
      <div className="architecture-grid">
        <div className="architecture-card">
          <div className="label">Dimensions d'embedding</div>
          <div className="architecture-controls">
            <button type="button" onClick={() => increment('embed_dim', -8, 8)}>-</button>
            <div className="value">{embedDim}</div>
            <button type="button" onClick={() => increment('embed_dim', 8, 8)}>+</button>
          </div>
          <small>Ajuste la taille du vector space byte-level.</small>
        </div>
        <div className="architecture-card">
          <div className="label">Blocs Transformer</div>
          <div className="architecture-controls">
            <button type="button" onClick={() => increment('num_layers', -1, 1)}>-</button>
            <div className="value">{numLayers}</div>
            <button type="button" onClick={() => increment('num_layers', 1, 1)}>+</button>
          </div>
          <small>Ajoute ou retire des blocs attention + MLP.</small>
        </div>
        <div className="architecture-card">
          <div className="label">Têtes d'attention</div>
          <div className="architecture-controls">
            <button type="button" onClick={() => increment('num_heads', -1, 1)}>-</button>
            <div className="value">{numHeads}</div>
            <button type="button" onClick={() => increment('num_heads', 1, 1)}>+</button>
          </div>
          <small>Répartit la projection QKV sur plusieurs sous-espaces.</small>
        </div>
        <div className="architecture-card">
          <div className="label">FF hidden dim</div>
          <div className="architecture-controls">
            <button type="button" onClick={() => increment('ff_hidden_dim', -128, 128)}>-</button>
            <div className="value">{ffDim}</div>
            <button type="button" onClick={() => increment('ff_hidden_dim', 128, 128)}>+</button>
          </div>
          <small>Dimension interne de la MLP par bloc.</small>
        </div>
        <div className="architecture-card">
          <div className="label">LayerNorm dims</div>
          <div className="architecture-controls">
            <button type="button" onClick={() => increment('layernorm_dim', -8, 8)}>-</button>
            <div className="value">{layerNormDim}</div>
            <button type="button" onClick={() => increment('layernorm_dim', 8, 8)}>+</button>
          </div>
          <small>Normalisation finale des sorties du backbone.</small>
        </div>
        <div className="architecture-card">
          <div className="label">Head dims</div>
          <div className="architecture-controls">
            <button type="button" onClick={() => increment('head_dim', -8, 8)}>-</button>
            <div className="value">{headDim}</div>
            <button type="button" onClick={() => increment('head_dim', 8, 8)}>+</button>
          </div>
          <small>Dimension d'entrée du head de sortie avant logits.</small>
        </div>
        <div className="architecture-card">
          <div className="label">Taille vocabulaire</div>
          <div className="architecture-controls">
            <button type="button" onClick={() => increment('vocab_size', -1024, 256)}>-</button>
            <div className="value">{formatNumber(vocabSize)}</div>
            <button type="button" onClick={() => increment('vocab_size', 1024, 256)}>+</button>
          </div>
          <small>Dimension du tokenizer (BPE ou byte-pair).</small>
        </div>
      </div>
      <div className="architecture-preview">
        {Array.from({ length: Math.max(numLayers, 1) }, (_, idx) => (
          <div key={idx} className="architecture-chip">
            <span className="chip-title">Bloc {idx + 1}</span>
            <span className="chip-meta">{numHeads} têtes · FF {ffDim}</span>
          </div>
        ))}
      </div>
      <div className="architecture-stats">
        <span className="stat-pill">Paramètres ≈ {formatNumber(totalParams)}</span>
        <span className="stat-pill">Blocs {numLayers}</span>
        <span className="stat-pill">Têtes {numHeads}</span>
        <span className="stat-pill">LayerNorm {layerNormDim}</span>
        <span className="stat-pill">Head {headDim}</span>
        <span className="stat-pill">Vocab {formatNumber(vocabSize)}</span>
      </div>
      <TransformerFlow summary={summary} />
    </section>
  );
}

function TransformerFlow({ summary }) {
  if (!summary) return null;
  const config = summary.config || {};
  const numLayers = Number(config.num_layers || 0);
  const embedDim = Number(config.embed_dim || 0);
  const ffDim = Number(config.ff_hidden_dim || 0);
  const numHeads = Number(config.num_heads || 0);
  const layerNormDim = Number(config.layernorm_dim || embedDim || 0);
  const headDim = Number(config.head_dim || layerNormDim || embedDim || 0);
  const vocabSize = Number(config.vocab_size || 256) || 256;

  const headMeta = headDim !== layerNormDim
    ? `${layerNormDim} → ${headDim} → ${vocabSize}`
    : `${layerNormDim} → ${vocabSize}`;
  const tokenMeta = vocabSize > 256 ? `Vocab ${formatNumber(vocabSize)}` : 'Entrée byte-level';

  const stages = [
    { title: 'Tokens', meta: tokenMeta },
    { title: 'Embedding', meta: `${embedDim} dims` },
    { title: `Blocs x${numLayers}`, meta: `${numHeads} têtes · FF ${ffDim}` },
    { title: 'LayerNorm', meta: `${layerNormDim} dims` },
    { title: 'Head', meta: headMeta },
    { title: 'Logits', meta: 'Distribution de sortie' },
  ];

  return (
    <div className="flow-diagram">
      {stages.map((stage, idx) => (
        <React.Fragment key={stage.title}>
          <div className="flow-node">
            <div className="flow-title">{stage.title}</div>
            <div className="flow-meta">{stage.meta}</div>
          </div>
          {idx < stages.length - 1 ? <div className="flow-arrow">→</div> : null}
        </React.Fragment>
      ))}
    </div>
  );
}

function TrainingPanel({ job, config, onChange, onStart, onStop, disabled, extraDirOptions = [] }) {
  const status = job?.status || 'idle';
  const details = job?.details || {};
  const lastEvent = details.last_event || '-';
  const step = details.step ?? '-';
  const trainLoss = details.train_loss;
  const valLoss = details.val_loss;
  const runDir = details.run_dir || '-';
  const message = job?.message;
  const selectedExtraDirs = Array.isArray(config.extra_data_dirs) ? config.extra_data_dirs : [];

  const toggleExtraDir = (path) => {
    const current = new Set(selectedExtraDirs);
    if (current.has(path)) {
      current.delete(path);
    } else {
      current.add(path);
    }
    onChange('extra_data_dirs', Array.from(current));
  };

  return (
    <section>
      <h2>Configuration & lancement de l'entraînement</h2>
      <div className="small">
        Le corpus par défaut englobe désormais <code>data_clean/</code> au complet. Ajustez les hyperparamètres ci-dessous puis lancez l'entraînement.
      </div>
      <div className="architecture-grid training-grid">
        {TRAINING_FIELDS.map((field) => (
          <div key={field.key} className="architecture-card training-card">
            <label className="label" htmlFor={`training-${field.key}`}>
              {field.label}
            </label>
            {field.type === 'select' && Array.isArray(field.options) ? (
              <select
                id={`training-${field.key}`}
                value={config[field.key] ?? ''}
                onChange={(event) => onChange(field.key, event.target.value)}
              >
                {field.options.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            ) : (
              <input
                id={`training-${field.key}`}
                type={field.type}
                inputMode={field.type === 'number' ? 'decimal' : undefined}
                step={field.step}
                min={field.min}
                placeholder={field.placeholder || field.label}
                value={config[field.key] ?? ''}
                onChange={(event) => onChange(field.key, event.target.value)}
              />
            )}
            {field.helper ? <small>{field.helper}</small> : null}
          </div>
        ))}
      </div>
      {extraDirOptions.length ? (
        <div className="architecture-card training-card" style={{ marginTop: '1rem' }}>
          <label className="label">Sources additionnelles</label>
          <div className="extra-data-grid">
            {extraDirOptions.map((option) => {
              const checked = selectedExtraDirs.includes(option.path);
              return (
                <label key={option.path} className={`checkbox ${option.available ? '' : 'checkbox-disabled'}`}>
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={disabled || !option.available}
                    onChange={() => toggleExtraDir(option.path)}
                  />
                  <span>
                    {option.label}
                    <span className="extra-data-meta">
                      {option.available ? formatBytes(option.sizeBytes) : 'Aucune donnée'} · {option.path}
                    </span>
                  </span>
                </label>
              );
            })}
          </div>
          <small>Par défaut toutes les sources nettoyées sont incluses. Cochez pour forcer l’ajout explicite de dossiers supplémentaires.</small>
        </div>
      ) : null}
      <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'space-between', gap: '0.75rem' }}>
        <button
          type="button"
          onClick={onStop}
          disabled={disabled || status !== 'running'}
          className="ghost-button"
        >
          {status === 'running' ? 'Arrêter et sauvegarder' : 'Arrêt indisponible'}
        </button>
        <button type="button" onClick={onStart} disabled={disabled || status === 'running'}>
          {status === 'running' ? 'Entraînement en cours...' : 'Démarrer l\'entraînement'}
        </button>
      </div>
      <div className="small" style={{ marginTop: '0.75rem' }}>
        <strong>Statut:</strong> {status}
        {message ? ` · ${message}` : ''}
        <br />
        <strong>Dernier évènement:</strong> {lastEvent}
        <br />
        <strong>Step:</strong> {step} · <strong>Train loss:</strong>{' '}
        {typeof trainLoss === 'number' ? trainLoss.toFixed(4) : '-'} · <strong>Val loss:</strong>{' '}
        {typeof valLoss === 'number' ? valLoss.toFixed(4) : '-'}
        <br />
        <strong>Run courant:</strong> {runDir}
      </div>
    </section>
  );
}

function buildSummaryFromConfig(config) {
  if (!config) return null;
  const embedDim = Number(config.embed_dim) || 0;
  const numLayers = Number(config.num_layers) || 0;
  const numHeads = Number(config.num_heads) || 0;
  const ffHiddenDim = Number(config.ff_hidden_dim) || 0;
  const blockSize = Number(config.block_size) || 0;
  const dropout = Number(config.dropout) || 0;
  const layerNormDim = Number(config.layernorm_dim) || embedDim;
  const headDim = Number(config.head_dim) || layerNormDim;
  const vocabSize = Number(config.vocab_size) || 256;

  const embeddingParams = vocabSize * embedDim;
  const layerNormProjParams = layerNormDim === embedDim ? 0 : embedDim * layerNormDim + layerNormDim;
  const layerNormFinalParams = layerNormProjParams + layerNormDim * 2;
  const headProjParams = headDim === layerNormDim ? 0 : layerNormDim * headDim + headDim;
  const headParams = headDim * vocabSize + headProjParams;

  const attentionWeights = embedDim * embedDim * 3;
  const attentionBias = embedDim * 3;
  const outputProjWeights = embedDim * embedDim;
  const outputProjBias = embedDim;
  const attentionParams = attentionWeights + attentionBias + outputProjWeights + outputProjBias;

  const ffWeights = embedDim * ffHiddenDim + ffHiddenDim * embedDim;
  const ffBias = ffHiddenDim + embedDim;
  const layerNormParams = embedDim * 4;
  const perLayerParams = attentionParams + ffWeights + ffBias + layerNormParams;

  const totalTransformerParams = perLayerParams * numLayers;
  const totalParams = embeddingParams + headParams + layerNormFinalParams + totalTransformerParams;

  return {
    total_params: totalParams,
    trainable_params: totalParams,
    blocks: [
      { name: 'Embedding', params: embeddingParams, trainable: embeddingParams },
      { name: 'LayerNorm', params: layerNormFinalParams, trainable: layerNormFinalParams },
      { name: 'Head', params: headParams, trainable: headParams },
    ],
    encoder_layers: Array.from({ length: numLayers }, (_, index) => ({
      name: `Bloc ${index + 1}`,
      params: perLayerParams,
      trainable: perLayerParams,
    })),
    config: {
      arch_preset: config.arch_preset || '',
      embed_dim: embedDim,
      layernorm_dim: layerNormDim,
      num_layers: numLayers,
      num_heads: numHeads,
      ff_hidden_dim: ffHiddenDim,
      block_size: blockSize,
      dropout,
      head_dim: headDim,
      vocab_size: vocabSize,
    },
  };
}

function MetricsPanel({ metrics }) {
  const { train = [], val = [] } = metrics || {};
  const combined = train.concat(val);
  const traces = [];
  if (train.length) {
    traces.push({
      x: train.map((item) => item.step),
      y: train.map((item) => item.loss),
      mode: 'lines+markers',
      name: 'train',
      line: { color: '#2563eb' },
      marker: { size: 6 },
    });
  }
  if (val.length) {
    traces.push({
      x: val.map((item) => item.step),
      y: val.map((item) => item.loss),
      mode: 'lines+markers',
      name: 'val',
      line: { color: '#f97316', dash: 'dot' },
      marker: { size: 6 },
    });
  }
  return (
    <section>
      <h2>Loss (train / val)</h2>
      {combined.length ? (
        <Plot
          data={traces}
          layout={{
            margin: { l: 60, r: 20, t: 30, b: 60 },
            xaxis: { title: 'Step', rangemode: 'tozero' },
            yaxis: { title: 'Loss', rangemode: 'tozero' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.2 },
            height: 360,
          }}
          config={{ displaylogo: false, responsive: true }}
          className="plot-container"
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      ) : (
        <div className="small">En attente de nouvelles mesures...</div>
      )}
      <div className="metrics-log" role="log">
        {combined.length
          ? combined
              .slice(-14)
              .reverse()
              .map((line) => formatMetric(line))
              .join('\n')
          : 'En attente de nouvelles mesures...'}
      </div>
    </section>
  );
}

function formatMetric(line) {
  const date = new Date(line.timestamp * 1000).toLocaleTimeString();
  const split = line.split === 'train' ? 'TRAIN' : 'VAL ';
  return `${date} | ${split} | step=${line.step} | loss=${line.loss.toFixed(4)}`;
}

// Lightweight t-SNE implementation adapted from Karpathy's tsnejs (MIT License)
function runTsne2D(data, options = {}) {
  const { perplexity = 20, iterations = 400, epsilon = 200 } = options;
  if (!Array.isArray(data) || data.length === 0) {
    throw new Error('Données t-SNE manquantes.');
  }
  const dim = Array.isArray(data[0]) ? data[0].length : 0;
  if (!dim) {
    throw new Error('Dimension des données invalide pour t-SNE.');
  }

  const tsne = new SimpleTSNE({ dim: 2, perplexity, epsilon });
  tsne.initDataRaw(data);
  const total = Math.max(10, iterations);
  for (let i = 0; i < total; i += 1) {
    tsne.step();
  }
  return tsne.getSolution();
}

function zeros(length) {
  return Array.from({ length }, () => 0);
}

function randn(mu = 0, std = 1) {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const mag = Math.sqrt(-2.0 * Math.log(u));
  return mu + std * mag * Math.cos(2.0 * Math.PI * v);
}

class SimpleTSNE {
  constructor(opts = {}) {
    this.perplexity = opts.perplexity || 30;
    this.dim = opts.dim || 2;
    this.epsilon = opts.epsilon || 10;
    this.iterations = 0;
    this.Y = null;
    this._momentum = 0.5;
    this._runningAvg = 0;
  }

  initDataRaw(data) {
    const n = data.length;
    if (this.perplexity * 3 > n) {
      throw new Error('Perplexité trop élevée pour la taille du jeu de données.');
    }
    this.N = n;
    this._distances = this._computeDistances(data);
    this.P = this._nearestNeighbourDistances(this._distances, this.perplexity);
    this.P = this._symmetrizeMatrix(this.P);
    this.P = this._normalizeMatrix(this.P);
    this.Y = Array.from({ length: this.N }, () => Array.from({ length: this.dim }, () => randn(0, 1e-4)));
    this._gradients = Array.from({ length: this.N }, () => zeros(this.dim));
    this._previousUpdates = Array.from({ length: this.N }, () => zeros(this.dim));
  }

  step() {
    if (!this.Y) return;
    const Q = this._computeQ(this.Y);
    const PQ = this._matrixDifference(this.P, Q.matrix);

    for (let i = 0; i < this.N; i += 1) {
      const grad = this._gradients[i];
      grad.fill(0);
      for (let j = 0; j < this.N; j += 1) {
        if (i === j) continue;
        const mult = PQ.get(i, j) * Q.norm(i, j);
        for (let d = 0; d < this.dim; d += 1) {
          grad[d] += mult * (this.Y[i][d] - this.Y[j][d]);
        }
      }
    }

    this.iterations += 1;
    const momentum = this.iterations < 250 ? this._momentum : 0.8;
    const eta = this.epsilon;

    for (let i = 0; i < this.N; i += 1) {
      for (let d = 0; d < this.dim; d += 1) {
        const grad = this._gradients[i][d];
        const update = momentum * this._previousUpdates[i][d] - eta * grad;
        this._previousUpdates[i][d] = update;
        this.Y[i][d] += update;
      }
    }
  }

  getSolution() {
    return this.Y || [];
  }

  _computeDistances(data) {
    const n = data.length;
    const D = Array.from({ length: n }, () => zeros(n));
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        let d = 0;
        const xi = data[i];
        const xj = data[j];
        for (let k = 0; k < xi.length; k += 1) {
          const diff = xi[k] - xj[k];
          d += diff * diff;
        }
        D[i][j] = d;
        D[j][i] = d;
      }
    }
    return D;
  }

  _nearestNeighbourDistances(distances, perplexity) {
    const tol = 1e-5;
    const n = distances.length;
    const P = Array.from({ length: n }, () => zeros(n));

    for (let i = 0; i < n; i += 1) {
      let betaMin = -Infinity;
      let betaMax = Infinity;
      let beta = 1;
      const row = distances[i].slice();
      row[i] = 0;

      let done = false;
      let attempts = 0;
      let currPerplexity = 0;
      while (!done && attempts < 50) {
        attempts += 1;
        let sumP = 0;
        for (let j = 0; j < n; j += 1) {
          if (i === j) {
            P[i][j] = 0;
          } else {
            P[i][j] = Math.exp(-row[j] * beta);
            sumP += P[i][j];
          }
        }
        if (sumP === 0) {
          sumP = 1e-12;
        }
        let entropy = 0;
        for (let j = 0; j < n; j += 1) {
          if (P[i][j] === 0) continue;
          const pij = P[i][j] / sumP;
          P[i][j] = pij;
          entropy -= pij * Math.log(pij + 1e-12);
        }
        currPerplexity = Math.exp(entropy);
        const perpDiff = currPerplexity - perplexity;
        if (Math.abs(perpDiff) < tol) {
          done = true;
        } else if (perpDiff > 0) {
          betaMin = beta;
          beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2;
        } else {
          betaMax = beta;
          beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2;
        }
      }
    }
    return P;
  }

  _symmetrizeMatrix(mat) {
    const n = mat.length;
    const sym = Array.from({ length: n }, () => zeros(n));
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        sym[i][j] = mat[i][j] + mat[j][i];
      }
    }
    return sym;
  }

  _normalizeMatrix(mat) {
    const n = mat.length;
    let sum = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        if (i === j) continue;
        sum += mat[i][j];
      }
    }
    if (sum === 0) {
      return mat;
    }
    const target = mat.map((row) => row.slice());
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        target[i][j] = mat[i][j] / sum;
      }
    }
    return target;
  }

  _computeQ(Y) {
    const n = Y.length;
    const dist = Array.from({ length: n }, () => zeros(n));
    let sum = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        let d = 0;
        for (let k = 0; k < this.dim; k += 1) {
          const diff = Y[i][k] - Y[j][k];
          d += diff * diff;
        }
        const num = 1 / (1 + d);
        dist[i][j] = num;
        dist[j][i] = num;
        sum += 2 * num;
      }
    }
    const matrix = Array.from({ length: n }, () => zeros(n));
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        if (i === j) {
          matrix[i][j] = 0;
        } else {
          matrix[i][j] = dist[i][j] / sum;
        }
      }
    }
    return {
      matrix,
      norm: (i, j) => dist[i][j],
      get: (i, j) => matrix[i][j],
    };
  }

  _matrixDifference(A, B) {
    const n = A.length;
    return {
      get(i, j) {
        return (A[i][j] || 0) - (B[i][j] || 0);
      },
    };
  }
}

function ModelSummary({ metadata }) {
  const summary =
    metadata?.model_summary ||
    metadata?.model?.summary ||
    metadata?.run?.model_summary ||
    buildSummaryFromConfig(metadata?.run?.config);
  if (!summary) {
    return (
      <section>
        <h2>Architecture du transformer</h2>
        <div className="small">Aucune information de modèle disponible.</div>
      </section>
    );
  }

  const configEntries = Object.entries(summary.config || {});

  return (
    <section>
      <h2>Architecture du transformer</h2>
      <div className="grid two-columns">
        <div className="badge">
          Paramètres totaux
          <strong>{formatNumber(summary.total_params)}</strong>
        </div>
        <div className="badge">
          Paramètres entraînables
          <strong>{formatNumber(summary.trainable_params)}</strong>
        </div>
      </div>
      {configEntries.length ? (
        <div className="grid two-columns" style={{ marginTop: '1rem' }}>
          {configEntries.map(([key, value]) => (
            <div key={key}>
              <div className="label">{key.replace(/_/g, ' ')}</div>
              <div className="value">{String(value)}</div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function EmbeddingPanel({
  records,
  selectedIndex,
  onSelectIndex,
  followLatest,
  onToggleFollow,
}) {
  const latestIndex = records.length ? records.length - 1 : 0;
  const record = records[selectedIndex] || null;

  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [tsneState, setTsneState] = useState({ status: 'idle', coords: [], step: null, error: null });

  const playbackRef = useRef(selectedIndex);
  useEffect(() => {
    playbackRef.current = selectedIndex;
  }, [selectedIndex]);

  useEffect(() => {
    if (!isPlaying || !records.length) {
      return () => {};
    }
    const timer = setInterval(() => {
      const next = (playbackRef.current + 1) % records.length;
      playbackRef.current = next;
      onSelectIndex(next);
    }, 1200);
    return () => clearInterval(timer);
  }, [isPlaying, records.length, onSelectIndex]);

  const scatterData = useMemo(() => {
    if (!record) return [];
    const coords = record.embedding?.coords || [];
    if (!coords.length) return [];
    const x = [];
    const y = [];
    const z = [];
    const ids = record.embedding.token_ids || [];
    const tokenStrings = record.embedding.token_strings || [];
    const hoverData = ids.map((id, index) => {
      const label = tokenStrings[id] ?? tokenStrings[index] ?? `token ${id}`;
      return [`${id}`, label];
    });
    coords.forEach((row) => {
      x.push(row[0] || 0);
      y.push(row[1] || 0);
      z.push(row[2] ?? 0);
    });
    return [
      {
        type: 'scatter3d',
        mode: 'markers',
        x,
        y,
        z,
        text: hoverData.map(([, label]) => label),
        customdata: hoverData,
        marker: {
          size: 3,
          color: z.length ? z : x,
          colorscale: 'Viridis',
        },
        hovertemplate: 'token %{customdata[0]} · %{customdata[1]}<extra></extra>',
      },
    ];
  }, [record]);

  const varianceInfo = useMemo(() => {
    if (!record) return '-';
    const ratios = record.embedding?.variance_ratio || [];
    if (!ratios.length) return '-';
    return ratios.map((r, idx) => `PC${idx + 1}: ${(r * 100).toFixed(1)} %`).join(' · ');
  }, [record]);

  const handleRunTsne = useCallback(() => {
    if (!record || !record.embedding?.coords?.length) {
      setTsneState({ status: 'error', coords: [], step: null, error: 'Aucune donnée à projeter.' });
      return;
    }
    const coords = record.embedding.coords.map((row) => row.map((value) => Number(value)));
    const perplexity = Math.max(5, Math.min(30, Math.floor(coords.length / 3) || 20));
    setTsneState({ status: 'running', coords: [], step: record.step, error: null });

    const compute = () => {
      try {
        const solution = runTsne2D(coords, { perplexity, iterations: 400 });
        setTsneState({ status: 'done', coords: solution, step: record.step, error: null });
      } catch (error) {
        setTsneState({
          status: 'error',
          coords: [],
          step: record.step,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    };

    if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
      window.requestIdleCallback(() => compute());
    } else {
      setTimeout(() => compute(), 0);
    }
  }, [record]);

  const renderContent = (suffix) => {
    const isTsneOutdated =
      tsneState.step !== null && record && tsneState.step !== record.step && tsneState.status === 'done';
    return (
      <div className="grid two-columns">
        <div>
          {records.length ? (
            <>
              <div className="viz-controls">
                <label htmlFor={`embedding-step-${suffix}`}>Étape visualisée</label>
                <input
                  id={`embedding-step-${suffix}`}
                  className="range"
                  type="range"
                  min={0}
                  max={latestIndex}
                  value={selectedIndex}
                  disabled={!records.length}
                  onChange={(event) => onSelectIndex(Number(event.target.value))}
                />
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <input
                    type="checkbox"
                    checked={followLatest}
                    disabled={!records.length}
                    onChange={(event) => onToggleFollow(event.target.checked)}
                  />
                  Suivre la dernière étape
                </label>
                <div className="small">
                  Étape {record?.step ?? '-'} ·{' '}
                  {record ? new Date((record.timestamp || 0) * 1000).toLocaleTimeString() : '-'}
                </div>
              </div>
              {scatterData.length ? (
                <Plot
                  data={scatterData}
                  layout={{
                    margin: { l: 0, r: 0, t: 30, b: 0 },
                    scene: {
                      xaxis: { title: 'PC1' },
                      yaxis: { title: 'PC2' },
                      zaxis: { title: 'PC3' },
                    },
                    height: suffix === 'fullscreen' ? 520 : 420,
                  }}
                  config={{ displaylogo: false, responsive: true }}
                  className="plot-container tall"
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler
                />
              ) : (
                <div className="small">Visualisation en attente...</div>
              )}
              <div className="small">Variance expliquée&nbsp;: {varianceInfo}</div>
            </>
          ) : (
            <div className="small">Aucun enregistrement de visualisation pour ce run.</div>
          )}
        </div>
        <div>
          <h3>Extraits enregistrés</h3>
          <div className="sample-log" role="log">
            {records.length ? (
              records
                .slice()
                .reverse()
                .filter((item) => item.sample)
                .map((item) => (
                  <div key={`${item.step}-${item.timestamp}`} className="sample-entry">
                    <div className="sample-meta">
                      Étape {item.step} ·{' '}
                      {item.timestamp ? new Date(item.timestamp * 1000).toLocaleTimeString() : '-'}
                    </div>
                    <pre>{item.sample}</pre>
                  </div>
                ))
            ) : (
              <div className="small">Aucun échantillon disponible pour l'instant.</div>
            )}
          </div>
          <div className="tsne-panel">
            <div className="tsne-header">
              <span>t-SNE expérimental</span>
              <button
                type="button"
                className="ghost-button"
                onClick={handleRunTsne}
                disabled={!records.length || tsneState.status === 'running'}
              >
                {tsneState.status === 'running' ? 'Calcul en cours…' : 'Lancer t-SNE'}
              </button>
            </div>
            {tsneState.status === 'error' ? (
              <div className="small warning">{tsneState.error}</div>
            ) : null}
            {tsneState.status === 'done' && tsneState.coords.length ? (
              <>
                {isTsneOutdated ? (
                  <div className="small warning">Résultat calculé pour l'étape {tsneState.step}. Relancez pour l'étape courante.</div>
                ) : null}
                <Plot
                  data={[
                    {
                      type: 'scatter',
                      mode: 'markers',
                      x: tsneState.coords.map((row) => row[0]),
                      y: tsneState.coords.map((row) => row[1]),
                      marker: { size: 5, color: '#f97316' },
                    },
                  ]}
                  layout={{
                    height: suffix === 'fullscreen' ? 420 : 320,
                    margin: { l: 40, r: 10, t: 30, b: 40 },
                    title: tsneState.step ? `t-SNE – étape ${tsneState.step}` : 't-SNE',
                  }}
                  config={{ displaylogo: false, responsive: true }}
                  className="plot-container"
                />
              </>
            ) : tsneState.status === 'running' ? (
              <div className="small">Calcul en arrière-plan…</div>
            ) : (
              <div className="small">Cliquez sur «&nbsp;Lancer t-SNE&nbsp;» pour projeter l'embedding actuel.</div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <section>
      <div className="section-header">
        <h2>Projection PCA des embeddings</h2>
        <div className="section-actions">
          <button
            type="button"
            className="ghost-button"
            onClick={() => setIsPlaying((prev) => !prev)}
            disabled={!records.length}
          >
            {isPlaying ? 'Pause' : 'Lecture'}
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={handleRunTsne}
            disabled={!records.length || tsneState.status === 'running'}
          >
            {tsneState.status === 'running' ? 't-SNE…' : 'Lancer t-SNE'}
          </button>
          <button type="button" onClick={() => setIsFullscreen(true)} disabled={!records.length}>
            Plein écran
          </button>
        </div>
      </div>
      {renderContent('inline')}
      {isFullscreen &&
        createPortal(
          <div className="fullscreen-overlay">
            <div className="fullscreen-content">
              <div className="section-header">
                <h2>Projection PCA des embeddings</h2>
                <div className="section-actions">
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => setIsPlaying((prev) => !prev)}
                    disabled={!records.length}
                  >
                    {isPlaying ? 'Pause' : 'Lecture'}
                  </button>
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={handleRunTsne}
                    disabled={!records.length || tsneState.status === 'running'}
                  >
                    {tsneState.status === 'running' ? 't-SNE…' : 'Relancer t-SNE'}
                  </button>
                  <button type="button" onClick={() => setIsFullscreen(false)}>
                    Fermer
                  </button>
                </div>
              </div>
              {renderContent('fullscreen')}
            </div>
          </div>,
          document.body,
        )}
    </section>
  );
}

function LogitPanel({ records }) {
  const latest = records.length ? records[records.length - 1] : null;
  const lineData = useMemo(() => {
    if (!records.length) return [];
    const steps = records.map((item) => item.step);
    const entropy = records.map((item) => item.logits?.entropy ?? null);
    const maxProb = records.map((item) => item.logits?.max_probability ?? null);
    return [
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Entropie',
        x: steps,
        y: entropy,
        line: { color: '#6366f1' },
        marker: { size: 6 },
        yaxis: 'y1',
      },
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Max probabilité',
        x: steps,
        y: maxProb,
        line: { color: '#f59e0b', dash: 'dot' },
        marker: { size: 6 },
        yaxis: 'y2',
      },
    ];
  }, [records]);

  const topTokens = useMemo(() => {
    if (!latest?.logits) return [];
    const ids = latest.logits.token_ids || [];
    const probs = latest.logits.probabilities || [];
    return ids.map((id, idx) => ({ id, prob: probs[idx] ?? 0 }));
  }, [latest]);

  return (
    <section>
      <h2>Évolution des logits</h2>
      {records.length ? (
        <Plot
          data={lineData}
          layout={{
            margin: { l: 60, r: 60, t: 30, b: 60 },
            xaxis: { title: 'Step' },
            yaxis: { title: 'Entropie' },
            yaxis2: {
              title: 'Max probabilité',
              overlaying: 'y',
              side: 'right',
              rangemode: 'tozero',
            },
            legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.2 },
            height: 320,
          }}
          config={{ displaylogo: false, responsive: true }}
          className="plot-container"
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      ) : (
        <div className="small">En attente de logits...</div>
      )}
      <div className="token-strip">
        {topTokens.length ? (
          topTokens.map(({ id, prob }) => (
            <span key={id} className="token-pill">
              token {id} · {(prob * 100).toFixed(1)} %
            </span>
          ))
        ) : (
          <span className="small">En attente de logits...</span>
        )}
      </div>
    </section>
  );
}

function ChatPanel({ defaults, onSend }) {
  const [context, setContext] = useState('');
  const [prompt, setPrompt] = useState('');
  const [maxTokens, setMaxTokens] = useState(defaults.max_new_tokens ?? 120);
  const [temperature, setTemperature] = useState(defaults.temperature ?? 0.8);
  const [topK, setTopK] = useState(defaults.top_k ?? 50);
  const [status, setStatus] = useState('Tapez un message pour lancer la génération.');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setMaxTokens(defaults.max_new_tokens ?? 120);
    setTemperature(defaults.temperature ?? 0.8);
    setTopK(defaults.top_k ?? 50);
  }, [defaults]);

  const send = async () => {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      setStatus('Veuillez saisir un message.');
      return;
    }
    setIsLoading(true);
    setStatus('Génération en cours...');
    try {
      const response = await onSend({
        context,
        prompt: trimmedPrompt,
        max_new_tokens: Number(maxTokens) || defaults.max_new_tokens || 120,
        temperature: Number(temperature) || defaults.temperature || 0.8,
        top_k: Number(topK) || defaults.top_k || 50,
      });
      const baseContext = context.trimEnd();
      const updatedContext = `${baseContext}${baseContext ? '\n' : ''}Utilisateur: ${trimmedPrompt}\nAssistant: ${response.generated_text || ''}`;
      setContext(updatedContext);
      setPrompt('');
      setStatus(response.generated_text || '(réponse vide)');
    } catch (err) {
      setStatus(`Erreur: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section>
      <h2>Conversation avec le modèle</h2>
      <div className="grid two-columns">
        <div>
          <label className="label" htmlFor="chat-context">Historique</label>
          <textarea
            id="chat-context"
            value={context}
            onChange={(event) => setContext(event.target.value)}
            placeholder={'Assistant: Bonjour !\nUtilisateur: Salut, comment puis-je t\'utiliser ?\n'}
          />
        </div>
        <div>
          <label className="label" htmlFor="chat-prompt">Message utilisateur</label>
          <textarea
            id="chat-prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Posez votre question ici..."
            onKeyDown={(event) => {
              if (event.key === 'Enter' && event.ctrlKey) {
                event.preventDefault();
                send();
              }
            }}
          />
        </div>
      </div>
      <div className="grid two-columns" style={{ marginTop: '1rem' }}>
        <div>
          <label className="label" htmlFor="max-tokens">Max tokens générés</label>
          <input
            id="max-tokens"
            type="number"
            min="1"
            max="2048"
            value={maxTokens}
            onChange={(event) => setMaxTokens(event.target.value)}
          />
        </div>
        <div>
          <label className="label" htmlFor="temperature">Température</label>
          <input
            id="temperature"
            type="number"
            step="0.05"
            min="0.05"
            max="2"
            value={temperature}
            onChange={(event) => setTemperature(event.target.value)}
          />
        </div>
        <div>
          <label className="label" htmlFor="top-k">Top-k</label>
          <input
            id="top-k"
            type="number"
            min="0"
            max="256"
            value={topK}
            onChange={(event) => setTopK(event.target.value)}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'flex-end' }}>
          <button type="button" onClick={send} disabled={isLoading}>
            {isLoading ? 'Génération...' : 'Envoyer'}
          </button>
        </div>
      </div>
      <div className="chat-output" role="status" aria-live="polite" style={{ marginTop: '1rem' }}>
        {status}
      </div>
    </section>
  );
}

function App() {
  const [metadata, setMetadata] = useState(null);
  const [metrics, setMetrics] = useState({ train: [], val: [] });
  const [visuals, setVisuals] = useState([]);
  const [selectedVisualIndex, setSelectedVisualIndex] = useState(0);
  const [followLatest, setFollowLatest] = useState(true);
  const [runSelectValue, setRunSelectValue] = useState('');
  const [statusMessage, setStatusMessage] = useState('Chargement des métadonnées...');
  const [jobs, setJobs] = useState({ data: {}, training: null, tokenizer: null });
  const [trainingConfig, setTrainingConfig] = useState({ ...DEFAULT_TRAINING_CONFIG });
  const [seedSource, setSeedSource] = useState(null);

  const [tokenizerConfig, setTokenizerConfig] = useState({ ...DEFAULT_TOKENIZER_CONFIG });
  const [tokenizerSeedSource, setTokenizerSeedSource] = useState(null);

  const [selectedTokenizerPath, setSelectedTokenizerPath] = useState('');
  const [tokenizerTestInput, setTokenizerTestInput] = useState('');
  const [tokenizerTestResult, setTokenizerTestResult] = useState(null);
  const [isTestingTokenizer, setIsTestingTokenizer] = useState(false);
  const [pendingRun, setPendingRun] = useState(null);
  const [newRunPending, setNewRunPending] = useState(false);
  const [availablePresets, setAvailablePresets] = useState({});

  const handleTokenizerTestInput = (value) => {
    setTokenizerTestInput(value);
    setTokenizerTestResult(null);
  };

  const defaults = metadata?.defaults || {
    max_new_tokens: 120,
    temperature: 0.8,
    top_k: 50,
  };
  const tokenizerDefaults = metadata?.tokenizer_defaults || {
    name: 'wiki-bpe',
    input_dir: 'data_clean',
    output_dir: 'trained_models/tokenizers',
    vocab_size: 32000,
    min_frequency: 2,
    lowercase: false,
  };
  const runs = metadata?.runs || [];
  const runOptions = useMemo(() => {
    if (pendingRun) {
      const exists = runs.some((run) => run.run_name === pendingRun.run_name || run.path === pendingRun.path);
      if (!exists) {
        return [pendingRun, ...runs];
      }
    }
    return runs;
  }, [pendingRun, runs]);
  const currentRunPath = metadata?.current_run_path || null;
  const trainingPresets = availablePresets;
  const availableTokenizers = metadata?.tokenizers || [];
  const resumeDisabled = jobs.training?.status === 'running';
  const canResumeRun = Boolean(metadata?.checkpoint_path);
  const extraDirOptions = useMemo(() => {
    const catalog = metadata?.data_catalog || {};
    return DATA_TASKS
      .map((task) => {
        const path = DATA_DIR_PATHS[task.key];
        if (!path || task.key === 'subtitles') return null;
        const stats = catalog[task.key] || {};
        const sizeBytes = Number(stats.size_bytes || 0);
        return {
          key: task.key,
          label: task.title,
          path,
          sizeBytes,
          available: Number.isFinite(sizeBytes) && sizeBytes > 0,
        };
      })
      .filter(Boolean);
  }, [metadata]);

  useEffect(() => {
    if (availableTokenizers.length) {
      const first = availableTokenizers[0];
      const defaultPath = first.tokenizer_path || first.path || '';
      setSelectedTokenizerPath((prev) => {
        if (!prev) return defaultPath;
        const exists = availableTokenizers.some((item) => {
          const pathCandidate = item.tokenizer_path || item.path;
          return pathCandidate && pathCandidate === prev;
        });
        return exists ? prev : defaultPath;
      });
    }
  }, [availableTokenizers]);

  const seedTrainingConfig = useCallback(
    (config) => {
      if (!config) return;
      setTrainingConfig((prev) => {
        const next = { ...prev };
        Object.entries(config).forEach(([key, value]) => {
          if (key in next && value !== null && value !== undefined) {
            if (Array.isArray(next[key])) {
              if (Array.isArray(value)) {
                next[key] = value.map((item) => String(item));
              } else if (typeof value === 'string') {
                next[key] = value
                  .split(',')
                  .map((item) => item.trim())
                  .filter(Boolean);
              } else {
                next[key] = [];
              }
              return;
            }
            if (INT_FIELDS.has(key)) {
              const parsed = Number(value);
              if (Number.isFinite(parsed)) {
                if (parsed > 0 || INT_FIELDS_ALLOW_ZERO.has(key)) {
                  next[key] = String(parsed);
                }
              }
              return;
            }
            if (FLOAT_FIELDS.has(key)) {
              const parsed = Number(value);
              if (Number.isFinite(parsed)) {
                next[key] = String(parsed);
              }
              return;
            }
            next[key] = String(value);
          }
        });
        ['max_steps', 'batch_size', 'block_size', 'num_layers', 'num_heads', 'ff_hidden_dim', 'embed_dim', 'vocab_size', 'eval_interval', 'eval_batches', 'sample_max_new_tokens'].forEach((key) => {
          if (INT_FIELDS.has(key)) {
            const current = Number(next[key]);
            if (!Number.isFinite(current) || current <= 0) {
              next[key] = DEFAULT_TRAINING_CONFIG[key];
            }
          }
        });
        return next;
      });
    },
    []
  );

  const seedTokenizerConfig = useCallback(
    (config) => {
      if (!config) return;
      setTokenizerConfig((prev) => {
        const next = { ...prev };
        Object.entries(config).forEach(([key, value]) => {
          if (key in next && value !== null && value !== undefined) {
            next[key] = typeof value === 'boolean' ? value : String(value);
          }
        });
        return next;
      });
    },
    []
  );

  const applyPreset = useCallback(
    (presetKey) => {
      setTrainingConfig((prev) => {
        const next = { ...prev };
        if (!presetKey) {
          next.arch_preset = '';
          return next;
        }
        next.arch_preset = presetKey;
        const preset = trainingPresets[presetKey];
        if (preset) {
          Object.entries(preset).forEach(([key, value]) => {
            if (value !== undefined && value !== null) {
              next[key] = String(value);
            }
          });
          const embed = preset.embed_dim ?? Number(next.embed_dim);
          if (embed) {
            next.embed_dim = String(embed);
            next.layernorm_dim = String(embed);
            next.head_dim = String(embed);
          }
          if (preset.vocab_size) {
            setTokenizerConfig((prevTokenizer) => ({
              ...prevTokenizer,
              vocab_size: String(preset.vocab_size),
            }));
          }
        }
        return next;
      });
    },
    [trainingPresets, setTokenizerConfig]
  );

  const handleNewRun = useCallback(() => {
    const runName = generateSlug('run');
    const tokenizerName = generateSlug('tokenizer');
    const pendingPath = `pending://${runName}`;
    const defaultPresetKey = availablePresets['very-small']
      ? 'very-small'
      : Object.keys(availablePresets)[0] || '';

    setTrainingConfig({
      ...DEFAULT_TRAINING_CONFIG,
      run_name: runName,
      arch_preset: defaultPresetKey,
    });
    setSeedSource(MANUAL_LOCK);
    setTokenizerConfig({ ...DEFAULT_TOKENIZER_CONFIG, name: tokenizerName });
    setTokenizerSeedSource(MANUAL_LOCK);
    setSelectedTokenizerPath('');
    setTokenizerTestInput('');
    setTokenizerTestResult(null);
    setIsTestingTokenizer(false);
    setPendingRun({ path: pendingPath, run_name: runName, status: 'pending' });
    setRunSelectValue(pendingPath);
    setMetadata(null);
    setJobs((prev) => ({ ...prev, training: null }));
    setMetrics({ train: [], val: [] });
    setVisuals([]);
    setSelectedVisualIndex(0);
    setFollowLatest(true);
    setNewRunPending(true);
    setStatusMessage(`Nouveau run prêt : ${runName}`);
    if (defaultPresetKey) {
      applyPreset(defaultPresetKey);
    }
  }, [availablePresets, applyPreset]);

  const loadMetadata = useCallback(async () => {
    try {
      const payload = await fetchJson('/metadata');
      const runsList = payload.runs || [];
      let match = null;
      if (pendingRun) {
        match = runsList.find((run) => {
          if (pendingRun.run_name && run.run_name === pendingRun.run_name) return true;
          if (pendingRun.run_name && run.path && run.path.includes(pendingRun.run_name)) return true;
          return false;
        });
        if (match) {
          setPendingRun(null);
          setNewRunPending(false);
          setRunSelectValue(match.path);
        }
      }
      if (newRunPending && pendingRun && !match) {
        setJobs({
          data: payload.jobs?.data || {},
          training: null,
          tokenizer: payload.jobs?.tokenizer || null,
        });
        return;
      }
      setMetadata({ ...payload, model_summary: payload.model_summary });
      setJobs({
        data: payload.jobs?.data || {},
        training: payload.jobs?.training || null,
        tokenizer: payload.jobs?.tokenizer || null,
      });
      if (payload.training_presets) {
        setAvailablePresets(payload.training_presets);
      }
      if (!pendingRun && !newRunPending && payload.current_run_path) {
        setRunSelectValue(payload.current_run_path);
      }
      setStatusMessage(
        payload.checkpoint_loaded === false
          ? 'Checkpoint non disponible.'
          : `Run courant : ${payload.run?.run_name || 'inconnu'}`
      );
      const trainingLocked = seedSource === MANUAL_LOCK;
      if (payload.run?.config && !trainingLocked) {
        const sourceKey = payload.current_run_path || payload.run?.run_name || 'default';
        if (!seedSource || seedSource !== sourceKey) {
          seedTrainingConfig(payload.run.config);
          setSeedSource(sourceKey);
        }
      }
      const tokenizerLocked = tokenizerSeedSource === MANUAL_LOCK;
      if (payload.tokenizer_defaults && !tokenizerLocked && !tokenizerSeedSource) {
        seedTokenizerConfig(payload.tokenizer_defaults);
        setTokenizerSeedSource('defaults');
      }
    } catch (err) {
      setStatusMessage(`Impossible de charger les métadonnées: ${err.message}`);
    }
  }, [seedSource, seedTrainingConfig, tokenizerSeedSource, seedTokenizerConfig, newRunPending, pendingRun]);

  const loadMetrics = useCallback(async () => {
    if (newRunPending || (runSelectValue && runSelectValue.startsWith('pending://'))) {
      setMetrics({ train: [], val: [] });
      return;
    }
    try {
      const payload = await fetchJson('/metrics');
      setMetrics({
        train: payload.train || [],
        val: payload.val || [],
      });
    } catch (err) {
      /* ignore */
    }
  }, [newRunPending, runSelectValue]);

  const loadVisuals = useCallback(async () => {
    if (newRunPending || (runSelectValue && runSelectValue.startsWith('pending://'))) {
      setVisuals([]);
      return;
    }
    try {
      const payload = await fetchJson('/visuals');
      const records = payload.records || [];
      setVisuals(records);
    } catch (err) {
      /* ignore */
    }
  }, [newRunPending, runSelectValue]);

  useEffect(() => {
    loadMetadata();
    loadMetrics();
    loadVisuals();
  }, [loadMetadata, loadMetrics, loadVisuals]);

  useEffect(() => {
    const interval = setInterval(loadMetadata, METADATA_REFRESH_MS);
    return () => clearInterval(interval);
  }, [loadMetadata]);

  useEffect(() => {
    const interval = setInterval(loadMetrics, LOSS_REFRESH_MS);
    return () => clearInterval(interval);
  }, [loadMetrics]);

  useEffect(() => {
    const interval = setInterval(loadVisuals, VISUAL_REFRESH_MS);
    return () => clearInterval(interval);
  }, [loadVisuals]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const status = await fetchJson('/tokenizer/status');
        setJobs((prev) => ({ ...prev, tokenizer: status }));
      } catch (err) { /* ignore */ }
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!visuals.length) return;
    if (followLatest) {
      setSelectedVisualIndex(visuals.length - 1);
    } else if (selectedVisualIndex > visuals.length - 1) {
      setSelectedVisualIndex(Math.max(visuals.length - 1, 0));
    }
  }, [visuals, followLatest, selectedVisualIndex]);

  const handleRunChange = async (event) => {
    const value = event.target.value;
    setRunSelectValue(value);
    if (!value) return;
    setSeedSource(null);
    setTokenizerSeedSource(null);
    setSelectedTokenizerPath('');
    setPendingRun(null);
    setNewRunPending(false);
    try {
      const payload = await fetchJson('/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run: value }),
      });
      setStatusMessage(
        payload.message
          ? payload.message
          : payload.checkpoint_loaded === false
          ? 'Checkpoint non disponible, en attente...'
          : `Run sélectionné : ${payload.selected}`
      );
      await Promise.all([loadMetadata(), loadMetrics(), loadVisuals()]);
    } catch (err) {
      setStatusMessage(`Erreur de sélection: ${err.message}`);
    }
  };

  const refreshRuns = async () => {
    try {
      const payload = await fetchJson('/runs');
      setMetadata((prev) => (prev ? { ...prev, runs: payload.runs || [] } : prev));
    } catch (err) {
      setStatusMessage(`Erreur rafraîchissement runs: ${err.message}`);
    }
  };

  const sendChat = async (body) => {
    const payload = await fetchJson('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return payload;
  };

  const visualRecords = useMemo(
    () =>
      visuals.map((item) => ({
        step: item.step,
        timestamp: item.timestamp,
        embedding: item.embedding,
        logits: item.logits,
      })),
    [visuals]
  );

  const handleTrainingConfigValue = (key, value) => {
    setTrainingConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleTokenizerConfigValue = (key, value) => {
    setTokenizerConfig((prev) => ({ ...prev, [key]: value }));
  };

  const startDataPreparation = async (task) => {
    if (task.disabled) return;
    const payload = { task: task.key };
    if (typeof task.buildPayload === 'function') {
      const extras = task.buildPayload();
      if (extras === null) {
        return;
      }
      if (typeof extras === 'object') {
        Object.assign(payload, extras);
      }
    } else if (task.params && typeof task.params === 'object') {
      Object.assign(payload, task.params);
    }
    try {
      await fetchJson('/data/prepare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      setStatusMessage(`Préparation des données « ${task.title} » lancée.`);
      await loadMetadata();
    } catch (err) {
      setStatusMessage(`Erreur préparation données: ${err.message}`);
    }
  };

  const startTokenizerTraining = async () => {
    const payload = {};
    Object.entries(tokenizerConfig).forEach(([key, value]) => {
      if (value === '' || value === null || value === undefined) return;
      if (key === 'lowercase') {
        payload[key] = Boolean(value);
        return;
      }
      if (TOKENIZER_INT_FIELDS.has(key)) {
        const parsed = parseInt(value, 10);
        if (!Number.isNaN(parsed)) payload[key] = parsed;
        return;
      }
      if (TOKENIZER_FLOAT_FIELDS.has(key)) {
        const parsed = parseFloat(value);
        if (!Number.isNaN(parsed)) payload[key] = parsed;
        return;
      }
      payload[key] = value;
    });
    try {
      setTokenizerTestResult(null);
      setSelectedTokenizerPath('');
      await fetchJson('/tokenizer/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      setStatusMessage('Apprentissage du tokenizer lancé.');
      await loadMetadata();
      setTokenizerSeedSource(null);
    } catch (err) {
      setStatusMessage(`Erreur entraînement tokenizer: ${err.message}`);
    }
  };

  const testTokenizer = async () => {
    if (!tokenizerTestInput.trim()) {
      setTokenizerTestResult({ ok: false, message: 'Entrez un texte à tokeniser.' });
      return;
    }
    setIsTestingTokenizer(true);
    try {
      const payload = await fetchJson('/tokenizer/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: tokenizerTestInput,
          tokenizer_path: selectedTokenizerPath || undefined,
        }),
      });
      setTokenizerTestResult({ ok: true, data: payload });
      if (!selectedTokenizerPath && payload.tokenizer_path) {
        setSelectedTokenizerPath(payload.tokenizer_path);
      }
    } catch (err) {
      setTokenizerTestResult({ ok: false, message: err.message });
    } finally {
      setIsTestingTokenizer(false);
    }
  };

  const resumeTraining = async () => {
    if (jobs.training?.status === 'running') {
      setStatusMessage("Un entraînement est déjà en cours.");
      return;
    }
    const runPath = metadata?.current_run_path;
    if (!runPath) {
      setStatusMessage('Aucun run sélectionné à reprendre.');
      return;
    }
    const checkpointPath = metadata?.checkpoint_path;
    if (!checkpointPath) {
      setStatusMessage("Aucun checkpoint disponible pour ce run.");
      return;
    }
    const currentMaxSteps = metadata?.run?.config?.max_steps;
    const promptDefault = currentMaxSteps ? String(currentMaxSteps) : '';
    const input = window.prompt(
      'Nouvelle valeur de max_steps (doit être supérieure à l\'étape actuelle).',
      promptDefault,
    );
    if (input === null) return;
    const parsed = Number.parseInt(input, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      window.alert('Valeur invalide pour max_steps.');
      return;
    }
    try {
      await fetchJson('/train/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          run: runPath,
          checkpoint: checkpointPath,
          max_steps: parsed,
        }),
      });
      setStatusMessage(`Reprise de l'entraînement lancée (max_steps=${parsed}).`);
      setMetrics({ train: [], val: [] });
      setVisuals([]);
      setSelectedVisualIndex(0);
      setFollowLatest(true);
      await Promise.all([loadMetadata(), loadMetrics(), loadVisuals()]);
    } catch (err) {
      setStatusMessage(`Erreur reprise: ${err.message}`);
    }
  };

  const startTraining = async () => {
    const payload = {};
    Object.entries(trainingConfig).forEach(([key, value]) => {
      if (key === 'extra_data_dirs') {
        if (Array.isArray(value)) {
          const normalised = Array.from(
            new Set(
              value
                .map((item) => (typeof item === 'string' ? item.trim() : String(item || '').trim()))
                .filter((entry) => entry.length)
            )
          );
          payload[key] = normalised;
        }
        return;
      }
      if (value === '' || value === null || value === undefined) return;
      if (INT_FIELDS.has(key)) {
        const parsed = parseInt(value, 10);
        if (!Number.isNaN(parsed)) {
          if (INT_FIELDS_ALLOW_ZERO.has(key) || parsed > 0) {
            payload[key] = parsed;
          }
        }
        return;
      }
      if (FLOAT_FIELDS.has(key)) {
        const parsed = parseFloat(value);
        if (!Number.isNaN(parsed)) payload[key] = parsed;
        return;
      }
      payload[key] = value;
    });
    if (payload.device === 'auto') {
      delete payload.device;
    }
    try {
      setMetrics({ train: [], val: [] });
      setVisuals([]);
      setSelectedVisualIndex(0);
      setFollowLatest(true);
      await fetchJson('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: payload }),
      });
      const runLabel = payload.run_name ? `« ${payload.run_name} »` : '';
      setStatusMessage(`Entraînement ${runLabel} lancé.`.trim());
      await loadMetadata();
      setSeedSource(null);
      setTokenizerSeedSource(null);
      setPendingRun(null);
      setNewRunPending(false);
    } catch (err) {
      setStatusMessage(`Erreur entraînement: ${err.message}`);
    }
  };

  const stopTraining = async () => {
    try {
      await fetchJson('/train/stop', {
        method: 'POST',
      });
      setStatusMessage('Arrêt de l\'entraînement demandé. Sauvegarde en cours.');
      await loadMetadata();
    } catch (err) {
      setStatusMessage(`Impossible d'arrêter l'entraînement: ${err.message}`);
    }
  };

  return (
    <>
      <header>
        <div>
          <h1>Suivi de l'entraînement</h1>
          <div className="small">{statusMessage}</div>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <div>
            <label className="label" htmlFor="run-select">Run en cours</label>
            <select id="run-select" value={runSelectValue || ''} onChange={handleRunChange}>
              <option value="">Choisir un run</option>
              {runOptions.map((run) => (
                <option key={run.path} value={run.path} disabled={run.status === 'pending'}>
                  {run.status === 'pending' ? `${run.run_name} (nouveau)` : run.run_name || run.path}
                </option>
              ))}
              {currentRunPath && !runOptions.some((run) => run.path === currentRunPath) ? (
                <option value={currentRunPath}>{currentRunPath}</option>
              ) : null}
            </select>
          </div>
          <button type="button" onClick={handleNewRun} className="ghost-button">Nouveau run</button>
          <button type="button" onClick={refreshRuns}>Actualiser</button>
        </div>
      </header>
      <main>
        <DataPreparationPanel
          statuses={jobs.data}
          catalog={metadata?.data_catalog}
          history={metadata?.data_history}
          onStart={startDataPreparation}
        />
        <TokenizerPanel
          config={tokenizerConfig}
          job={jobs.tokenizer}
          onConfigChange={handleTokenizerConfigValue}
          onStart={startTokenizerTraining}
          onTest={testTokenizer}
          testing={isTestingTokenizer}
          testInput={tokenizerTestInput}
          onTestInputChange={handleTokenizerTestInput}
          existing={availableTokenizers}
          selectedTokenizer={selectedTokenizerPath}
          onSelectTokenizer={setSelectedTokenizerPath}
          testResult={tokenizerTestResult}
        />
        <ArchitectureBuilder
          config={trainingConfig}
          presets={trainingPresets}
          onConfigChange={handleTrainingConfigValue}
          onApplyPreset={applyPreset}
        />
        <TrainingPanel
          job={jobs.training}
          config={trainingConfig}
          onChange={handleTrainingConfigValue}
          onStart={startTraining}
          onStop={stopTraining}
          extraDirOptions={extraDirOptions}
        />
        <ModelSummary metadata={metadata} />
        <RunDetails
          metadata={metadata}
          onResume={canResumeRun ? resumeTraining : undefined}
          resumeDisabled={resumeDisabled}
        />
        <EmbeddingPanel
          records={visualRecords}
          selectedIndex={selectedVisualIndex}
          onSelectIndex={(index) => {
            setSelectedVisualIndex(index);
            setFollowLatest(false);
          }}
          followLatest={followLatest}
          onToggleFollow={(checked) => setFollowLatest(checked)}
        />
        <LogitPanel records={visualRecords.filter((item) => item.logits)} />
        <MetricsPanel metrics={metrics} />
        <ChatPanel defaults={defaults} onSend={sendChat} />
      </main>
    </>
  );
}

export default App;
