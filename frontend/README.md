# Declaration QA Frontend

Static frontend for the Declaration QA Workflow service.

Run it locally:

```bash
python -m http.server 8020 -d frontend
```

Default backend:

```text
http://127.0.0.1:8010
```

Use the backend independently:

```bash
python -m uvicorn declaration_qa.server:app --host 127.0.0.1 --port 8010 --reload
```

The frontend talks only to `/api/declaration-qa/*` endpoints.
