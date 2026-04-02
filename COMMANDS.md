# Project Nebula — Spis komend

## Stack Docker

```powershell
# Uruchom wszystko (pierwsze uruchomienie lub po zmianach kodu)
docker compose up --build -d

# Uruchom wszystko bez przebudowy (szybkie)
docker compose up -d

# Zatrzymaj wszystko (kontenery zostają, dane nie są usuwane)
docker compose stop

# Zatrzymaj i usuń kontenery (dane w wolumenach zostają)
docker compose down

# Zatrzymaj i usuń kontenery ORAZ dane bazy i mediów (RESET)
docker compose down -v

# Sprawdź status kontenerów
docker compose ps

# Logi wszystkich serwisów na żywo
docker compose logs -f

# Logi konkretnego serwisu
docker compose logs -f api_gateway
docker compose logs -f orchestrator
docker compose logs -f dsp_worker
docker compose logs -f video_worker
docker compose logs -f db_migrate
```

> **UWAGA: `db_migrate` zawsze pokazuje się jako "Stopped" — to normalne.**
> Jest to kontener jednorazowy (one-shot): uruchamia migracje Alembic, drukuje
> `[migrate] Schema up-to-date.` i kończy działanie z kodem 0. Nie powinien
> działać cały czas. Wszystkie pozostałe serwisy uruchamiają się dopiero po
> jego pomyślnym zakończeniu.

---

## Auto-start przy starcie Windows

Kontenery mają ustawione `restart: always` — uruchamiają się automatycznie
gdy Docker Desktop startuje. Upewnij się, że Docker Desktop ma włączone
**"Start Docker Desktop when you log in"**:

```
Docker Desktop → Settings → General → ✅ Start Docker Desktop when you log in
```

Po włączeniu tej opcji cały stack startuje automatycznie po uruchomieniu Windows.

---

## API — Generowanie miksu

```powershell
# Health check
Invoke-RestMethod http://localhost:8000/health

# Generuj miks (zwraca mix_id i task_id)
Invoke-RestMethod -Uri http://localhost:8000/mixes/generate `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"requested_duration_minutes": 45, "style_hint": "Dark Neurofunk"}'

# Generuj miks z konkretnym BPM
Invoke-RestMethod -Uri http://localhost:8000/mixes/generate `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"requested_duration_minutes": 60, "style_hint": "Liquid DnB", "force_bpm": 174}'

# Sprawdź status miksu (podmień <mix_id>)
Invoke-RestMethod http://localhost:8000/mixes/<mix_id>

# Pobierz gotowy miks jako plik WAV (działa gdy status = complete)
Invoke-WebRequest -Uri http://localhost:8000/mixes/<mix_id>/audio/download `
  -OutFile "nebula_mix.wav"
# Następnie otwórz nebula_mix.wav w dowolnym odtwarzaczu (Windows Media Player, VLC, itp.)
```

---

## Monitorowanie

```
# Flower — dashboard kolejek Celery (użytkownik: admin, hasło: z .env FLOWER_PASSWORD)
http://localhost:5555

# API docs (Swagger)
http://localhost:8000/docs

# API docs (ReDoc)
http://localhost:8000/redoc
```

---

## Testy

```powershell
# Testy schematów Pydantic (lokalnie, nie wymaga Dockera)
python -m pytest tests/unit/test_tasks/ -v

# Testy DSP audio (w kontenerze dsp_worker)
docker exec nebula_dsp_worker python -m pytest tests/unit/test_audio/ -v

# Wszystkie testy jednostkowe (w kontenerze orchestrator)
docker exec nebula_orchestrator python -m pytest tests/unit/ -v

# Testy z raportem coverage
docker exec nebula_dsp_worker python -m pytest tests/unit/test_audio/ -v --tb=short

# Testy integracyjne (wymaga działającego stacku: docker compose up -d)
python -m pytest tests/integration/ -v

# Testy integracyjne przeciwko innemu serwerowi
$env:API_BASE_URL="http://192.168.1.100:8000"; python -m pytest tests/integration/ -v
```

---

## Baza danych

```powershell
# Połącz się z PostgreSQL (wymaga psql lub klienta DB)
# Host: localhost, Port: 5432, DB: nebula_db, User: nebula

# Uruchom migracje ręcznie (normalnie robi to db_migrate automatycznie)
docker exec nebula_orchestrator alembic upgrade head

# Sprawdź aktualną wersję schematu
docker exec nebula_orchestrator alembic current

# Historia migracji
docker exec nebula_orchestrator alembic history
```

---

## Rebuild po zmianach kodu

```powershell
# Przebuduj i zrestartuj konkretny serwis
docker compose up --build -d api_gateway
docker compose up --build -d orchestrator
docker compose up --build -d dsp_worker
docker compose up --build -d video_worker

# Przebuduj wszystko
docker compose up --build -d
```

---

## Diagnostyka problemów

```powershell
# Sprawdź dlaczego kontener nie startuje
docker compose logs --tail=50 <nazwa_serwisu>

# Wejdź do kontenera interaktywnie
docker exec -it nebula_orchestrator bash
docker exec -it nebula_dsp_worker bash
docker exec -it nebula_api_gateway bash

# Sprawdź zmienne środowiskowe w kontenerze
docker exec nebula_orchestrator env | Select-String "POSTGRES\|REDIS\|OPENAI"

# Ping Redis z kontenera
docker exec nebula_orchestrator redis-cli -h redis ping

# Sprawdź połączenie z bazą
docker exec nebula_orchestrator python -c "from shared.db.session import _sync_engine; print('DB OK')"
```

---

## Naprawa błędu: `password authentication failed for user "Marksio"`

Błąd oznacza, że plik `.env` ma ustawione `POSTGRES_USER` na nazwę użytkownika Windows
zamiast `nebula`. Baza danych PostgreSQL nie zna użytkownika `Marksio`.

**Krok 1 — Sprawdź plik `.env`** (otwórz w notatniku lub VSCode):

```powershell
notepad .env
```

Upewnij się, że masz **dokładnie takie wartości**:

```env
POSTGRES_USER=nebula
POSTGRES_PASSWORD=nebula_secret
POSTGRES_DB=nebula_db
```

> Jeśli plik `.env` nie istnieje — skopiuj `.env.example` do `.env` i uzupełnij klucze API.

**Krok 2 — Sprawdź czy hasło zgadza się z wolumenem Postgres**

Jeśli Postgres był już zainicjalizowany z innym hasłem, samo poprawienie `.env` nie wystarczy.
Zresetuj wolumen bazy (UWAGA: usuwa wszystkie dane miksu!):

```powershell
docker compose down -v
docker compose up -d
```

**Krok 3 — Weryfikacja po naprawie**

```powershell
docker compose logs -f db_migrate
```

Poprawny wynik:

```
[migrate] Waiting for postgres...
[migrate] Schema up-to-date.
```
