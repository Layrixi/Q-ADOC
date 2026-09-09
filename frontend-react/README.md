# DocAnswerer frontend

React + Vite chat UI for the RAG backend (`app/main.py`).

## Setup

```bash
cd frontend-react
npm install
npm run dev
```

Opens at `http://localhost:5173`. Requires the FastAPI backend running separately:

```bash
uvicorn app.main:app --reload --port 8000
```

## Theming

All colors and fonts live in **`src/styles/variables.css`** — that's the single
place to change the look of the whole app. Every component references these
CSS custom properties rather than hardcoding values.

## Structure

```
src/
  components/     one component + its .css file per UI piece
  hooks/useApi.js  all backend fetch() calls in one place
  styles/          variables.css (tokens) + global.css (resets, a11y helpers)
  App.jsx          top-level layout + state
```

## Accessibility notes

- Built to WCAG 2.2 AA: verified color contrast ratios, visible keyboard
  focus (`:focus-visible`), skip link, semantic landmarks, `aria-live`
  regions for the connection status and upload feedback, `role="status"`
  on the loading/thinking indicator.
- Respects `prefers-reduced-motion` (disables the message fade-in and
  thinking-dot animation).
- Responsive: sidebar becomes an off-canvas drawer below 860px width.
