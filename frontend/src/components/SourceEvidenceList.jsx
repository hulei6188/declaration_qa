import React, { useMemo, useState } from 'react';
import { ChevronDown, ExternalLink, FileText, Image } from 'lucide-react';
import { API_BASE } from '../utils.js';

export default function SourceEvidenceList({ sources = [] }) {
  const items = useMemo(() => normalizeSources(sources), [sources]);
  const [activeIndex, setActiveIndex] = useState(null);

  if (!items.length) return null;

  const activeSource = activeIndex === null ? null : items[activeIndex];

  return (
    <div className="answer-sources" aria-label="证据来源">
      <div className="answer-sources-title">证据来源</div>
      <div className="answer-source-chips">
        {items.map((source, index) => {
          const image = sourceImage(source);
          const active = activeIndex === index;
          return (
            <button
              key={sourceKey(source, index)}
              type="button"
              className={`answer-source-chip${active ? ' active' : ''}`}
              title={fullSourceLabel(source)}
              onClick={() => setActiveIndex(active ? null : index)}
            >
              {image ? <Image size={13} /> : <FileText size={13} />}
              <span>{index + 1}</span>
              <strong>{shortSourceLabel(source)}</strong>
              <ChevronDown size={12} />
            </button>
          );
        })}
      </div>
      {activeSource && <SourceDetail source={activeSource} />}
    </div>
  );
}

function SourceDetail({ source }) {
  const image = sourceImage(source);
  const facts = sourceFacts(source);
  const link = sourceLink(source);

  return (
    <div className="answer-source-detail">
      {image && (
        <a className="answer-source-media" href={image.original} target="_blank" rel="noreferrer" title="查看原图">
          <img src={image.thumbnail} alt={sourceImageAlt(source)} loading="lazy" />
          <span>
            <ExternalLink size={12} />
            查看原图
          </span>
        </a>
      )}
      <div className="answer-source-detail-main">
        <div className="answer-source-detail-head">
          <strong>{source.title || '未命名来源'}</strong>
          {link && (
            <a href={link} target="_blank" rel="noreferrer">
              <ExternalLink size={12} />
              打开来源
            </a>
          )}
        </div>
        {facts.length > 0 && (
          <div className="answer-source-facts">
            {facts.map((fact) => (
              <span key={fact}>{fact}</span>
            ))}
          </div>
        )}
        <p>{source.snippet || '暂无证据摘要'}</p>
        <code>score={formatScore(source.score)} · {source.retrieval_channel || '-'} · {source.region_code || '-'} · {source.policy_version || '-'}</code>
      </div>
    </div>
  );
}

function normalizeSources(sources) {
  const seen = new Set();
  const output = [];
  for (const source of sources || []) {
    const key = sourceKey(source, output.length);
    if (seen.has(key)) continue;
    seen.add(key);
    output.push(source);
  }
  return output.slice(0, 12);
}

function sourceKey(source, index) {
  return source?.chunk_id || `${source?.doc_id || 'source'}-${source?.title || ''}-${index}`;
}

function sourceLabel(source) {
  const title = source?.title || sourceFilename(source) || '未命名来源';
  const section = sourceSection(source);
  return [title, section].filter(Boolean).join(' · ');
}

function fullSourceLabel(source) {
  return sourceLabel(source);
}

function shortSourceLabel(source) {
  const metadata = source?.metadata || {};
  const parts = sectionParts(source);
  const leaf = (
    metadata.node_title ||
    metadata.section_title ||
    parts.at(-1) ||
    metadata.attachment_name ||
    sourceFilename(source) ||
    source?.title ||
    '来源'
  );
  const parent = parts.length >= 2 ? parts.at(-2) : '';
  const label = parent && parent !== leaf ? `${parent} / ${leaf}` : leaf;
  return compactLabel(label, 18);
}

function sourceSection(source) {
  const parts = sectionParts(source);
  if (parts.length) return parts.slice(-2).join(' > ');
  const metadata = source?.metadata || {};
  return (
    metadata.node_title ||
    metadata.section_title ||
    metadata.attachment_name ||
    metadata.source_attachment_chunk_type ||
    firstSnippetLine(source?.snippet)
  );
}

function sectionParts(source) {
  const metadata = source?.metadata || {};
  const headingPath = metadata.heading_path || metadata.section_path;
  if (Array.isArray(headingPath)) {
    return headingPath.map((item) => cleanPathPart(item)).filter(Boolean);
  }
  if (typeof headingPath === 'string' && headingPath.trim()) {
    return headingPath.split(/[>／/]+/).map((item) => cleanPathPart(item)).filter(Boolean);
  }
  const sectionTitle = cleanPathPart(metadata.section_title);
  const nodeTitle = cleanPathPart(metadata.node_title);
  return [sectionTitle, nodeTitle].filter(Boolean);
}

function cleanPathPart(value) {
  return String(value || '')
    .replace(/办事指南/g, '')
    .replace(/&gt;/g, '>')
    .replace(/[#|*`>-]/g, '')
    .trim();
}

function compactLabel(value, maxLength) {
  const text = String(value || '').replace(/\s+/g, '');
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength)}…`;
}

function firstSnippetLine(value) {
  return String(value || '')
    .split('\n')
    .map((line) => line.replace(/[#|*`>-]/g, '').trim())
    .find(Boolean) || '';
}

function sourceFilename(source) {
  const metadata = source?.metadata || {};
  return metadata.asset_filename || metadata.attachment_name || metadata.source_file || '';
}

function isImageSource(source) {
  const metadata = source?.metadata || {};
  const hasAssetUrl = Boolean(metadata.asset_url || metadata.thumbnail_url);
  const imageLike = metadata.modality === 'image' || Boolean(metadata.asset_kind);
  return hasAssetUrl && imageLike;
}

function sourceImage(source) {
  if (!isImageSource(source)) return null;
  const metadata = source?.metadata || {};
  return {
    thumbnail: absoluteAssetUrl(metadata.thumbnail_url || metadata.asset_url || ''),
    original: absoluteAssetUrl(metadata.asset_url || metadata.thumbnail_url || ''),
  };
}

function sourceLink(source) {
  const image = sourceImage(source);
  if (image?.original) return image.original;
  const metadata = source?.metadata || {};
  const url = source?.source_url || metadata.source_url || metadata.official_source_url || '';
  if (!url || /^internal:\/\//i.test(url)) return '';
  return absoluteAssetUrl(url);
}

function absoluteAssetUrl(url) {
  if (!url) return '';
  if (/^https?:\/\//i.test(url)) return url;
  return `${API_BASE}${url}`;
}

function sourceImageAlt(source) {
  return sourceFilename(source) || source?.title || '证据图片';
}

function sourceFacts(source) {
  const metadata = source?.metadata || {};
  const facts = [];
  if (isImageSource(source)) facts.push(assetKindLabel(metadata.asset_kind));
  const pageNo = metadata.attachment_page_no || metadata.page_no;
  const pageRange = metadata.attachment_page_range;
  if (pageNo) facts.push(`第 ${pageNo} 页`);
  else if (pageRange) facts.push(`页码 ${pageRange}`);
  const role = roleLabel(metadata.attachment_role);
  if (role) facts.push(role);
  const filename = sourceFilename(source);
  if (filename) facts.push(filename);
  return facts.filter(Boolean).slice(0, 5);
}

function assetKindLabel(value) {
  return {
    pdf_page_image: 'PDF 页面',
    docx_embedded_image: 'DOCX 图片',
    original_image: '图片附件',
  }[value] || (value ? String(value) : '图片资产');
}

function roleLabel(value) {
  return {
    blank_form: '空表',
    sample_file: '样表',
    flowchart: '流程图',
    other_attachment: '附件',
  }[value] || '';
}

function formatScore(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) return '-';
  return number.toFixed(4);
}
