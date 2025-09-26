from decouple import config
import psycopg2, os

def build_dsn():
    host = config('SUPABASE_DB_HOST', default=None)
    if host:
        name = config('SUPABASE_DB_NAME')
        user = config('SUPABASE_DB_USER')
        pwd = config('SUPABASE_DB_PASSWORD')
        port = config('SUPABASE_DB_PORT', default='5432')
        dsn = f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
        print(f"Using SUPABASE_* DSN -> {host}:{port}")
        return dsn
    dsn = config('DATABASE_URL', default=None)
    if dsn:
        print("Using DATABASE_URL")
        return dsn
    raise SystemExit("DATABASE_URL or SUPABASE_* vars are missing.")

try:
    dsn = build_dsn()
    conn = psycopg2.connect(dsn, sslmode='require')
    print('✅ Database connection successful!')
    conn.close()
except Exception as e:
    print('❌ Connection failed:', e)