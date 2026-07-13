-- Panel 4 (CRUD) — tabla de consultas guardadas del predictor Liga 1 Perú
-- Correr una sola vez en el SQL Editor de Supabase.

create table if not exists consultas (
  id                bigint generated always as identity primary key,
  created_at        timestamptz not null default now(),
  equipo_local      text not null,
  equipo_visitante  text not null,
  fecha_partido     text,
  prob_xg           numeric,
  prob_tiros        numeric,
  prob_goles        numeric,
  alto_xg           boolean,
  alto_tiros        boolean,
  alto_goles        boolean,
  nota              text
);

-- RLS: proyecto academico sin sistema de usuarios, se habilita acceso
-- publico total via la key "anon"/"publishable" (nunca la service_role).
alter table consultas enable row level security;

create policy "Acceso publico total (proyecto academico sin auth)"
  on consultas
  for all
  to anon
  using (true)
  with check (true);
