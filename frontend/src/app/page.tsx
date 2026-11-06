"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity, Brain, Check, ChevronDown, CircleAlert, Clock3,
  Database, FileSearch, FlaskConical, LayoutDashboard, Loader2,
  Menu, Microscope, RefreshCw, Search, ShieldCheck, Sparkles,
  Users, X, XCircle, ArrowUpRight, Bot, Network, BookOpen,
} from "lucide-react";
import { api } from "@/lib/api";

type Section =
  | "Dashboard"
  | "AI Analysis"
  | "Signals"
  | "Human Review"
  | "Sponsors"
  | "Agent Memory";

const AGENTS = [
  "broken_promises_agent",
  "missing_results_agent",
  "track_record_agent",
  "pattern_finder_agent",
  "side_effect_agent",
  "timeline_agent",
];

const NAV = [
  { name: "Dashboard", icon: LayoutDashboard },
  { name: "AI Analysis", icon: FlaskConical },
  { name: "Signals", icon: FileSearch },
  { name: "Human Review", icon: ShieldCheck },
  { name: "Sponsors", icon: Users },
  { name: "Agent Memory", icon: Brain },
] as const;

const cx = (...classes: (string | false | null | undefined)[]) =>
  classes.filter(Boolean).join(" ");

function asArray(value: any, key?: string): any[] {
  if (Array.isArray(value)) return value;
  if (key && Array.isArray(value?.[key])) return value[key];
  return [];
}

function pretty(value: unknown) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function shortAgent(value: string) {
  return value.replace("_agent", "").replaceAll("_", " ");
}

function formatDate(value: any) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function confidence(value: any) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return `${Math.round((n <= 1 ? n : n / 100) * 100)}%`;
}

function StatusPill({ value }: { value: any }) {
  const status = String(value || "unknown").toLowerCase();
  const style =
    status.includes("approved") || status === "approve"
      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
      : status.includes("pending")
      ? "border-amber-500/30 bg-amber-500/10 text-amber-300"
      : status.includes("reject")
      ? "border-rose-500/30 bg-rose-500/10 text-rose-400"
      : "border-white/10 bg-white/5 text-slate-300";

  return (
    <span className={cx("inline-flex rounded-full border px-2.5 py-1 text-xs capitalize", style)}>
      {status.replaceAll("_", " ")}
    </span>
  );
}

function Metric({
  label,
  value,
  icon: Icon,
  note,
}: {
  label: string;
  value: string | number;
  icon: any;
  note?: string;
}) {
  return (
    <div className="rounded-2xl border border-white/[0.08] bg-[#11121b] p-5">
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">{label}</p>
        <div className="rounded-xl border border-purple-400/20 bg-purple-500/10 p-2 text-purple-300">
          <Icon size={18} />
        </div>
      </div>
      <p className="mt-4 text-3xl font-semibold tracking-tight text-white">{value}</p>
      {note && <p className="mt-2 text-xs text-slate-500">{note}</p>}
    </div>
  );
}

function Panel({
  title,
  subtitle,
  action,
  children,
}: {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-white/[0.08] bg-[#11121b]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.07] px-5 py-4">
        <div>
          <h3 className="font-semibold text-white">{title}</h3>
          {subtitle && <p className="mt-1 text-xs text-slate-500">{subtitle}</p>}
        </div>
        {action}
      </div>
      <div className="p-5">{children}</div>
    </section>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="rounded-xl border border-dashed border-white/10 px-4 py-10 text-center text-sm text-slate-500">
      {text}
    </div>
  );
}

function ErrorText({ error }: { error: string }) {
  return (
    <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-3 text-sm text-rose-300">
      {error}
    </div>
  );
}

export default function Home() {
  const [section, setSection] = useState<Section>("Dashboard");
  const [health, setHealth] = useState<any>(null);
  const [signals, setSignals] = useState<any[]>([]);
  const [reviewData, setReviewData] = useState<any>(null);
  const [episodesData, setEpisodesData] = useState<any>(null);
  const [sponsorsData, setSponsorsData] = useState<any>(null);

  const [sponsorLookup, setSponsorLookup] = useState<any>(null);
  const [sponsorLookupLoading, setSponsorLookupLoading] = useState(false);
  const [sponsorLookupError, setSponsorLookupError] = useState("");

  const [loading, setLoading] = useState(true);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [search, setSearch] = useState("");
  const [selectedSignal, setSelectedSignal] = useState<any>(null);
  const [selectedSponsor, setSelectedSponsor] = useState<any>(null);
  const [selectedAgent, setSelectedAgent] = useState(AGENTS[0]);
  const [proceduresData, setProceduresData] = useState<any>(null);

  const [analysisTask, setAnalysisTask] = useState(
    "Analyze clinical trials for research integrity signals, including missing results, safety reporting gaps, sponsor track record, and timeline inconsistencies."
  );
  const [nctIds, setNctIds] = useState("");
  const [maxStudies, setMaxStudies] = useState("10");
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);

  const [actionLoading, setActionLoading] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [reviewReason, setReviewReason] = useState("");
  const [mobileNav, setMobileNav] = useState(false);
  const [notice, setNotice] = useState("");

  const loadData = useCallback(async () => {
    setLoading(true);
    const nextErrors: Record<string, string> = {};

    const tasks = await Promise.allSettled([
      api.health(),
      api.signals("limit=100"),
      api.reviewQueue(),
      api.episodes("limit=20"),
      api.sponsors(),
    ]);

    const setters = [
      (v: any) => setHealth(v),
      (v: any) => setSignals(asArray(v)),
      (v: any) => setReviewData(v),
      (v: any) => setEpisodesData(v),
      (v: any) => setSponsorsData(v),
    ];

    const names = ["health", "signals", "review", "episodes", "sponsors"];

    tasks.forEach((result, index) => {
      if (result.status === "fulfilled") {
        setters[index](result.value);
      } else {
        nextErrors[names[index]] =
          result.reason?.message || `Could not load ${names[index]}`;
      }
    });

    setErrors(nextErrors);
    setLoading(false);
  }, []);

  const loadProcedures = useCallback(async (agent: string) => {
    try {
      const result = await api.procedures(agent);
      setProceduresData(result);
    } catch (error: any) {
      setProceduresData(null);
      setErrors((old) => ({ ...old, procedures: error.message }));
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (section === "Agent Memory") loadProcedures(selectedAgent);
  }, [section, selectedAgent, loadProcedures]);

  // Search the loaded sponsor list and also look up the exact sponsor
  // through the existing individual sponsor API endpoint.
  useEffect(() => {
    if (section !== "Sponsors") return;

    const term = search.trim();

    if (!term) {
      setSponsorLookup(null);
      setSponsorLookupError("");
      setSponsorLookupLoading(false);
      return;
    }

    let active = true;

    const timer = window.setTimeout(async () => {
      setSponsorLookupLoading(true);
      setSponsorLookupError("");

      try {
        const result = await api.sponsor(term);

        if (active) {
          setSponsorLookup(result);
        }
      } catch (error: any) {
        if (active) {
          setSponsorLookup(null);
          setSponsorLookupError(error.message || "Sponsor not found.");
        }
      } finally {
        if (active) setSponsorLookupLoading(false);
      }
    }, 400);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [section, search]);

  const reviewQueue = asArray(reviewData, "queue");
  const episodes = asArray(episodesData, "episodes");
  const sponsors = asArray(sponsorsData, "sponsors");
  const details = health?.details || {};

  const filteredSignals = useMemo(() => {
    const term = search.toLowerCase().trim();

    if (!term) return signals;

    return signals.filter((item) =>
      [
        item.agent,
        item.signal_type,
        item.summary,
        item.nct_id,
        item.status,
      ]
        .join(" ")
        .toLowerCase()
        .includes(term)
    );
  }, [signals, search]);

  const filteredSponsors = useMemo(() => {
    const term = search.toLowerCase().trim();

    if (!term) return sponsors;

    const matches = sponsors.filter((item) =>
      JSON.stringify(item).toLowerCase().includes(term)
    );

    if (
      sponsorLookup &&
      typeof sponsorLookup === "object" &&
      !Array.isArray(sponsorLookup)
    ) {
      const lookupName = String(
        sponsorLookup.sponsor ||
        sponsorLookup.sponsor_name ||
        sponsorLookup.name ||
        ""
      ).toLowerCase();

      const alreadyExists = matches.some((item) =>
        String(item.sponsor || item.sponsor_name || item.name || "")
          .toLowerCase() === lookupName
      );

      if (!alreadyExists && lookupName) {
        matches.unshift(sponsorLookup);
      }
    }

    return matches;
  }, [sponsors, search, sponsorLookup]);

  async function runAnalysis(event: React.FormEvent) {
    event.preventDefault();
    if (!analysisTask.trim()) return;

    setAnalysisLoading(true);
    setAnalysisResult(null);
    setNotice("");

    try {
      const payload: any = {
        task: analysisTask.trim(),
        max_studies: Number(maxStudies) || 10,
      };

      const ids = nctIds
        .split(/[\n, ]+/)
        .map((id) => id.trim())
        .filter(Boolean);

      if (ids.length) payload.nct_ids = ids;

      const result = await api.analyze(payload);
      setAnalysisResult(result);
      setNotice("Analysis completed successfully.");
      await loadData();
    } catch (error: any) {
      setNotice(error.message || "Analysis failed.");
    } finally {
      setAnalysisLoading(false);
    }
  }

  async function submitReview(
    queueId: string,
    decision: "approve" | "reject" | "edit"
  ) {
    if (!reviewer.trim()) {
      setNotice("Enter a reviewer name before submitting a decision.");
      return;
    }

    if (decision === "reject" && !reviewReason.trim()) {
      setNotice("A rejection reason is required.");
      return;
    }

    if (decision === "edit" && !reviewReason.trim()) {
      setNotice("Enter the edit summary.");
      return;
    }

    setActionLoading(queueId + decision);
    setNotice("");

    try {
      const payload: any = {
        decision,
        reviewer: reviewer.trim(),
      };

      if (decision === "reject") {
        payload.rejection_reason = reviewReason.trim();
      }

      if (decision === "edit") {
        payload.edit_summary = reviewReason.trim();
      }

      const result = await api.review(queueId, payload);
      setNotice(result.message || `Review ${decision} submitted.`);
      setReviewReason("");
      await loadData();
    } catch (error: any) {
      setNotice(error.message || "Could not submit review.");
    } finally {
      setActionLoading("");
    }
  }

  async function openSignal(item: any) {
    setSelectedSignal(item);

    const id = item.signal_id;
    if (!id) return;

    try {
      const detail = await api.signal(String(id));
      setSelectedSignal(detail);
    } catch {
      // Keep the signal list record visible if detail retrieval fails.
    }
  }

  async function openSponsor(item: any) {
    setSelectedSponsor(item);

    const name = item.sponsor || item.sponsor_name || item.name;
    if (!name) return;

    try {
      const detail = await api.sponsor(String(name));
      setSelectedSponsor(detail);
    } catch {
      // Keep the sponsor list record visible if detail retrieval fails.
    }
  }

  const pageTitle = section;
  const healthStatus = health?.status || "unknown";

  return (
    <main className="min-h-screen bg-[#090a10] text-slate-200">
      <div className="flex min-h-screen">
        <aside
          className={cx(
            "fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-white/[0.07] bg-[#0d0e16] transition-transform lg:static lg:translate-x-0",
            mobileNav ? "translate-x-0" : "-translate-x-full"
          )}
        >
          <div className="flex h-20 items-center gap-3 border-b border-white/[0.07] px-6">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-purple-500 to-violet-800 text-white shadow-lg shadow-purple-950/50">
              <Microscope size={21} />
            </div>

            <div>
              <p className="text-lg font-bold tracking-[0.18em] text-white">
                MOSAIC
              </p>
              <p className="text-[10px] tracking-[0.2em] text-slate-500">
                RESEARCH INTELLIGENCE
              </p>
            </div>

            <button
              onClick={() => setMobileNav(false)}
              className="ml-auto text-slate-400 lg:hidden"
              aria-label="Close navigation"
            >
              <X size={19} />
            </button>
          </div>

          <div className="px-4 pt-7">
            <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-600">
              Workspace
            </p>

            <nav className="space-y-1">
              {NAV.map(({ name, icon: Icon }) => (
                <button
                  key={name}
                  onClick={() => {
                    setSection(name);
                    setMobileNav(false);
                    setSearch("");
                    setNotice("");
                  }}
                  className={cx(
                    "flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition",
                    section === name
                      ? "border border-purple-400/20 bg-purple-500/10 text-purple-200"
                      : "text-slate-400 hover:bg-white/[0.04] hover:text-white"
                  )}
                >
                  <Icon size={18} />
                  {name}

                  {name === "Human Review" &&
                    Number(reviewData?.total_pending) > 0 && (
                      <span className="ml-auto rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] text-amber-300">
                        {reviewData.total_pending}
                      </span>
                    )}
                </button>
              ))}
            </nav>
          </div>

          <div className="mt-auto border-t border-white/[0.07] p-5">
            <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span
                  className={cx(
                    "h-2 w-2 rounded-full",
                    healthStatus === "healthy"
                      ? "bg-emerald-400"
                      : "bg-rose-400"
                  )}
                />
                API {healthStatus}
              </div>

              <p className="mt-2 text-[11px] text-slate-600">
                MOSAIC · Version {health?.version || "—"}
              </p>
            </div>
          </div>
        </aside>

        {mobileNav && (
          <button
            className="fixed inset-0 z-30 bg-black/60 lg:hidden"
            onClick={() => setMobileNav(false)}
            aria-label="Close menu overlay"
          />
        )}

        <div className="min-w-0 flex-1">
          <header className="sticky top-0 z-20 flex h-20 items-center justify-between border-b border-white/[0.07] bg-[#090a10]/90 px-5 backdrop-blur-xl md:px-8">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setMobileNav(true)}
                className="rounded-lg border border-white/10 p-2 text-slate-300 lg:hidden"
                aria-label="Open navigation"
              >
                <Menu size={18} />
              </button>

              <div>
                <p className="text-sm font-semibold text-white">{pageTitle}</p>
                <p className="hidden text-xs text-slate-500 sm:block">
                  Clinical trial research integrity intelligence
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-2 text-xs text-slate-400 sm:flex">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                Live API connection
              </div>

              <button
                onClick={loadData}
                disabled={loading}
                className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-slate-300 transition hover:bg-white/[0.08] disabled:opacity-50"
              >
                <RefreshCw
                  size={14}
                  className={loading ? "animate-spin" : ""}
                />
                <span className="hidden sm:inline">Refresh</span>
              </button>
            </div>
          </header>

          <div className="mx-auto max-w-[1500px] space-y-7 p-5 md:p-8">
            {notice && (
              <div className="flex items-start justify-between gap-3 rounded-xl border border-purple-400/20 bg-purple-500/10 px-4 py-3 text-sm text-purple-200">
                {notice}
                <button
                  onClick={() => setNotice("")}
                  aria-label="Dismiss notice"
                >
                  <X size={16} />
                </button>
              </div>
            )}

            {section === "Dashboard" && (
              <>
                <div className="relative overflow-hidden rounded-3xl border border-purple-400/15 bg-gradient-to-br from-[#21143b] via-[#151426] to-[#10111b] p-7 md:p-9">
                  <div className="pointer-events-none absolute -right-16 -top-28 h-80 w-80 rounded-full bg-purple-500/15 blur-3xl" />

                  <div className="relative max-w-3xl">
                    <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-purple-300/20 bg-purple-400/10 px-3 py-1.5 text-xs text-purple-200">
                      <Sparkles size={13} />
                      Multi-agent intelligence system
                    </div>

                    <h1 className="text-3xl font-semibold tracking-tight text-white md:text-4xl">
                      Research integrity,{" "}
                      <span className="text-purple-300">in focus.</span>
                    </h1>

                    <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
                      MOSAIC screens clinical trial registry and literature
                      evidence for candidate research-integrity signals using
                      six specialist agents and human review.
                    </p>

                    <button
                      onClick={() => setSection("AI Analysis")}
                      className="mt-6 inline-flex items-center gap-2 rounded-xl bg-purple-500 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-purple-950/40 transition hover:bg-purple-400"
                    >
                      Start an analysis <ArrowUpRight size={16} />
                    </button>
                  </div>
                </div>

                {errors.health && <ErrorText error={errors.health} />}

                <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                  <Metric
                    label="Signals in database"
                    value={details.signals_in_db ?? "—"}
                    icon={FileSearch}
                    note="Stored candidate signals"
                  />
                  <Metric
                    label="Pending reviews"
                    value={details.pending_reviews ?? "—"}
                    icon={Clock3}
                    note="Awaiting analyst decision"
                  />
                  <Metric
                    label="Memory episodes"
                    value={details.episodes_count ?? "—"}
                    icon={Brain}
                    note="Recorded agent sessions"
                  />
                  <Metric
                    label="Review queue depth"
                    value={details.queue_depth ?? "—"}
                    icon={Activity}
                    note="Current queue size"
                  />
                </div>

                <div className="grid gap-5 xl:grid-cols-[1.15fr_0.85fr]">
                  <Panel
                    title="Recent signals"
                    subtitle="Latest records returned by the signals API"
                    action={
                      <button
                        onClick={() => setSection("Signals")}
                        className="text-xs text-purple-300 hover:text-purple-200"
                      >
                        View all{" "}
                        <ArrowUpRight className="ml-1 inline" size={13} />
                      </button>
                    }
                  >
                    {errors.signals ? (
                      <ErrorText error={errors.signals} />
                    ) : signals.length === 0 ? (
                      <Empty text="No signals returned by the API." />
                    ) : (
                      <div className="space-y-3">
                        {signals.slice(0, 5).map((item, i) => (
                          <button
                            key={item.signal_id || i}
                            onClick={() => openSignal(item)}
                            className="flex w-full items-start justify-between gap-4 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3 text-left hover:border-purple-400/20"
                          >
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium text-slate-200">
                                {item.summary || item.signal_type || "Signal record"}
                              </p>
                              <p className="mt-1 text-xs text-slate-500">
                                {shortAgent(item.agent || "Unknown agent")} ·{" "}
                                {item.nct_id || "No trial ID"}
                              </p>
                            </div>

                            <div className="shrink-0 text-right">
                              <p className="text-sm text-purple-300">
                                {confidence(item.confidence) || "—"}
                              </p>
                              <StatusPill value={item.status} />
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </Panel>

                  <Panel title="System status" subtitle="Live health endpoint">
                    <div className="flex items-center gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/[0.06] p-4">
                      <div className="rounded-xl bg-emerald-500/10 p-3 text-emerald-400">
                        <Activity size={22} />
                      </div>
                      <div>
                        <p className="font-semibold capitalize text-emerald-300">
                          {healthStatus}
                        </p>
                        <p className="mt-1 text-xs text-slate-500">
                          MOSAIC API · v{health?.version || "—"}
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 space-y-3 text-sm">
                      <div className="flex justify-between">
                        <span className="text-slate-500">Database</span>
                        <span className="text-slate-200">
                          {health?.database || "—"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Application</span>
                        <span className="text-slate-200">
                          {health?.app || "—"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">API version</span>
                        <span className="text-slate-200">
                          {health?.version || "—"}
                        </span>
                      </div>
                    </div>
                  </Panel>
                </div>

                <Panel
                  title="Specialist agent network"
                  subtitle="Six agents configured in the MOSAIC analysis workflow"
                >
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                    {AGENTS.map((agent, i) => (
                      <div
                        key={agent}
                        className="flex items-center gap-3 rounded-xl border border-white/[0.07] bg-white/[0.02] p-4"
                      >
                        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-purple-500/10 text-purple-300">
                          <Bot size={18} />
                        </div>
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium capitalize text-slate-200">
                            {shortAgent(agent)}
                          </p>
                          <p className="text-[11px] text-slate-500">
                            Specialist agent {String(i + 1).padStart(2, "0")}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </Panel>
              </>
            )}

            {section === "AI Analysis" && (
              <>
                <div>
                  <h1 className="text-2xl font-semibold text-white">
                    Run an analysis
                  </h1>
                  <p className="mt-2 text-sm text-slate-500">
                    Submit a task to the six-agent MOSAIC workflow.
                  </p>
                </div>

                <div className="grid gap-5 xl:grid-cols-[1fr_0.8fr]">
                  <Panel
                    title="Analysis configuration"
                    subtitle="Uses POST /api/v1/analyze"
                  >
                    <form onSubmit={runAnalysis} className="space-y-5">
                      <div>
                        <label className="mb-2 block text-sm text-slate-300">
                          Analysis task
                        </label>
                        <textarea
                          value={analysisTask}
                          onChange={(e) => setAnalysisTask(e.target.value)}
                          rows={5}
                          className="w-full rounded-xl border border-white/10 bg-[#0b0c13] p-4 text-sm text-slate-200 outline-none transition placeholder:text-slate-600 focus:border-purple-400/50"
                          placeholder="Describe what MOSAIC should investigate..."
                        />
                      </div>

                      <div>
                        <label className="mb-2 block text-sm text-slate-300">
                          Clinical trial IDs{" "}
                          <span className="text-slate-600">(optional)</span>
                        </label>
                        <textarea
                          value={nctIds}
                          onChange={(e) => setNctIds(e.target.value)}
                          rows={3}
                          className="w-full rounded-xl border border-white/10 bg-[#0b0c13] p-4 text-sm text-slate-200 outline-none focus:border-purple-400/50"
                          placeholder={"NCT00000000\nNCT00000001"}
                        />
                        <p className="mt-2 text-xs text-slate-600">
                          Separate multiple NCT IDs with commas, spaces or new
                          lines.
                        </p>
                      </div>

                      <div>
                        <label className="mb-2 block text-sm text-slate-300">
                          Maximum studies
                        </label>
                        <input
                          type="number"
                          min="1"
                          max="50"
                          value={maxStudies}
                          onChange={(e) => setMaxStudies(e.target.value)}
                          className="w-full rounded-xl border border-white/10 bg-[#0b0c13] px-4 py-3 text-sm text-slate-200 outline-none focus:border-purple-400/50"
                        />
                      </div>

                      <button
                        disabled={analysisLoading || !analysisTask.trim()}
                        className="flex w-full items-center justify-center gap-2 rounded-xl bg-purple-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-purple-400 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        {analysisLoading ? (
                          <>
                            <Loader2 className="animate-spin" size={17} />
                            Running six agents...
                          </>
                        ) : (
                          <>
                            <FlaskConical size={17} />
                            Run MOSAIC analysis
                          </>
                        )}
                      </button>
                    </form>
                  </Panel>

                  <Panel
                    title="Analysis response"
                    subtitle="Result returned by the backend"
                  >
                    {analysisLoading ? (
                      <div className="flex flex-col items-center py-14 text-center">
                        <Loader2
                          size={32}
                          className="animate-spin text-purple-300"
                        />
                        <p className="mt-4 text-sm text-slate-300">
                          MOSAIC agents are processing...
                        </p>
                        <p className="mt-2 text-xs text-slate-500">
                          This may take some time.
                        </p>
                      </div>
                    ) : analysisResult ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                          <div className="rounded-xl bg-white/[0.04] p-3">
                            <p className="text-xs text-slate-500">Total signals</p>
                            <p className="mt-1 text-2xl text-white">
                              {analysisResult.total_signals ?? "—"}
                            </p>
                          </div>
                          <div className="rounded-xl bg-white/[0.04] p-3">
                            <p className="text-xs text-slate-500">Needs review</p>
                            <p className="mt-1 text-2xl text-white">
                              {analysisResult.signals_requiring_review ?? "—"}
                            </p>
                          </div>
                        </div>

                        <div>
                          <p className="text-xs text-slate-500">Run ID</p>
                          <p className="mt-1 break-all font-mono text-xs text-purple-300">
                            {analysisResult.run_id}
                          </p>
                        </div>

                        <div>
                          <p className="text-xs text-slate-500">Duration</p>
                          <p className="mt-1 text-sm text-slate-200">
                            {analysisResult.duration_seconds ?? "—"} seconds
                          </p>
                        </div>

                        <div>
                          <p className="text-xs text-slate-500">
                            Agents activated
                          </p>
                          <p className="mt-1 text-sm text-slate-200">
                            {asArray(analysisResult.agents_activated).join(", ") || "—"}
                          </p>
                        </div>

                        <div>
                          <p className="text-xs text-slate-500">Final brief</p>
                          <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded-xl bg-[#0b0c13] p-4 text-xs leading-6 text-slate-300">
                            {analysisResult.final_brief || "No brief returned."}
                          </pre>
                        </div>
                      </div>
                    ) : (
                      <Empty text="Your analysis result will appear here after a run." />
                    )}
                  </Panel>
                </div>

                <p className="text-xs leading-5 text-slate-600">
                  MOSAIC surfaces candidate research-integrity signals for
                  human triage. A generated signal is not a determination of
                  misconduct.
                </p>
              </>
            )}

            {section === "Signals" && (
              <>
                <div className="flex flex-wrap items-end justify-between gap-4">
                  <div>
                    <h1 className="text-2xl font-semibold text-white">
                      Research signals
                    </h1>
                    <p className="mt-2 text-sm text-slate-500">
                      Candidate signals returned by the signals API.
                    </p>
                  </div>

                  <div className="relative w-full sm:w-72">
                    <Search
                      className="absolute left-3 top-3 text-slate-500"
                      size={16}
                    />
                    <input
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Search signals..."
                      className="w-full rounded-xl border border-white/10 bg-[#11121b] py-2.5 pl-9 pr-3 text-sm outline-none focus:border-purple-400/50"
                    />
                  </div>
                </div>

                {errors.signals ? (
                  <ErrorText error={errors.signals} />
                ) : (
                  <Panel
                    title={`${filteredSignals.length} signals`}
                    subtitle="Filter by agent, signal type, summary, NCT ID or status"
                  >
                    {filteredSignals.length === 0 ? (
                      <Empty text="No matching signals returned by the API." />
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="w-full min-w-[720px] text-left text-sm">
                          <thead className="text-xs text-slate-500">
                            <tr className="border-b border-white/[0.07]">
                              <th className="pb-3 font-medium">Signal</th>
                              <th className="pb-3 font-medium">Agent</th>
                              <th className="pb-3 font-medium">Trial ID</th>
                              <th className="pb-3 font-medium">Confidence</th>
                              <th className="pb-3 font-medium">Status</th>
                              <th className="pb-3 font-medium"></th>
                            </tr>
                          </thead>

                          <tbody>
                            {filteredSignals.map((item, i) => (
                              <tr
                                key={item.signal_id || i}
                                className="border-b border-white/[0.05] last:border-0"
                              >
                                <td className="max-w-sm py-4 pr-4">
                                  <p className="line-clamp-2 text-slate-200">
                                    {item.summary || item.signal_type || "Signal record"}
                                  </p>
                                  <p className="mt-1 text-[11px] text-slate-600">
                                    {formatDate(item.created_at)}
                                  </p>
                                </td>
                                <td className="py-4 pr-4 capitalize text-slate-400">
                                  {shortAgent(item.agent || "—")}
                                </td>
                                <td className="py-4 pr-4 font-mono text-xs text-slate-400">
                                  {item.nct_id || "—"}
                                </td>
                                <td className="py-4 pr-4 text-purple-300">
                                  {confidence(item.confidence) || "—"}
                                </td>
                                <td className="py-4 pr-4">
                                  <StatusPill value={item.status} />
                                </td>
                                <td className="py-4">
                                  <button
                                    onClick={() => openSignal(item)}
                                    className="text-xs text-purple-300 hover:text-purple-200"
                                  >
                                    Details
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </Panel>
                )}
              </>
            )}

            {section === "Human Review" && (
              <>
                <div>
                  <h1 className="text-2xl font-semibold text-white">
                    Human review queue
                  </h1>
                  <p className="mt-2 text-sm text-slate-500">
                    Review candidate signals and submit analyst decisions.
                  </p>
                </div>

                {errors.review && <ErrorText error={errors.review} />}

                <div className="grid gap-4 sm:grid-cols-3">
                  <Metric
                    label="Pending"
                    value={reviewData?.total_pending ?? "—"}
                    icon={Clock3}
                  />
                  <Metric
                    label="Approved"
                    value={reviewData?.total_approved ?? "—"}
                    icon={Check}
                  />
                  <Metric
                    label="Rejected"
                    value={reviewData?.total_rejected ?? "—"}
                    icon={XCircle}
                  />
                </div>

                <Panel
                  title="Pending items"
                  subtitle="Ordered by confidence ascending by the backend"
                >
                  {reviewQueue.length === 0 ? (
                    <Empty text="There are no pending review items returned by the API." />
                  ) : (
                    <div className="space-y-4">
                      {reviewQueue.map((item, i) => {
                        const queueId = String(
                          item.review_id || item.queue_id || item.id || ""
                        );
                        const signal = item.signal || item;

                        return (
                          <div
                            key={queueId || i}
                            className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4"
                          >
                            <div className="flex flex-wrap items-start justify-between gap-3">
                              <div>
                                <p className="text-sm font-semibold text-slate-200">
                                  {signal.summary || signal.signal_type || "Review item"}
                                </p>
                                <p className="mt-1 text-xs text-slate-500">
                                  {shortAgent(signal.agent || item.agent || "—")} ·{" "}
                                  {signal.nct_id || "No trial ID"}
                                </p>
                              </div>
                              <StatusPill
                                value={item.decision || signal.status || "pending"}
                              />
                            </div>

                            <div className="mt-3 grid gap-2 text-xs text-slate-400 sm:grid-cols-3">
                              <p>
                                Confidence:{" "}
                                <span className="text-purple-300">
                                  {confidence(signal.confidence) || "—"}
                                </span>
                              </p>
                              <p>
                                Queue ID:{" "}
                                <span className="font-mono text-slate-500">
                                  {queueId || "Not provided"}
                                </span>
                              </p>
                              <p>
                                Created:{" "}
                                {formatDate(item.created_at || signal.created_at)}
                              </p>
                            </div>

                            {signal.evidence && (
                              <pre className="mt-3 max-h-32 overflow-auto whitespace-pre-wrap rounded-lg bg-black/20 p-3 text-xs leading-5 text-slate-400">
                                {pretty(signal.evidence)}
                              </pre>
                            )}

                            <div className="mt-4 grid gap-3 md:grid-cols-2">
                              <input
                                value={reviewer}
                                onChange={(e) => setReviewer(e.target.value)}
                                placeholder="Reviewer name"
                                className="rounded-lg border border-white/10 bg-[#0b0c13] px-3 py-2.5 text-sm outline-none focus:border-purple-400/50"
                              />
                              <input
                                value={reviewReason}
                                onChange={(e) => setReviewReason(e.target.value)}
                                placeholder="Reason (required for reject/edit)"
                                className="rounded-lg border border-white/10 bg-[#0b0c13] px-3 py-2.5 text-sm outline-none focus:border-purple-400/50"
                              />
                            </div>

                            <div className="mt-3 flex flex-wrap gap-2">
                              <button
                                disabled={!queueId || !!actionLoading}
                                onClick={() => submitReview(queueId, "approve")}
                                className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-300 hover:bg-emerald-500/20 disabled:opacity-40"
                              >
                                {actionLoading === queueId + "approve"
                                  ? "Submitting..."
                                  : "Approve"}
                              </button>
                              <button
                                disabled={!queueId || !!actionLoading}
                                onClick={() => submitReview(queueId, "reject")}
                                className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-xs text-rose-300 hover:bg-rose-500/20 disabled:opacity-40"
                              >
                                {actionLoading === queueId + "reject"
                                  ? "Submitting..."
                                  : "Reject"}
                              </button>
                              <button
                                disabled={!queueId || !!actionLoading}
                                onClick={() => submitReview(queueId, "edit")}
                                className="rounded-lg border border-purple-500/20 bg-purple-500/10 px-3 py-2 text-xs text-purple-300 hover:bg-purple-500/20 disabled:opacity-40"
                              >
                                {actionLoading === queueId + "edit"
                                  ? "Submitting..."
                                  : "Submit edit"}
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </Panel>
              </>
            )}

            {section === "Sponsors" && (
              <>
                <div className="flex flex-wrap items-end justify-between gap-4">
                  <div>
                    <h1 className="text-2xl font-semibold text-white">
                      Sponsor profiles
                    </h1>
                    <p className="mt-2 text-sm text-slate-500">
                      Sponsor credibility profiles available in semantic memory.
                    </p>
                  </div>

                  <div className="relative w-full sm:w-72">
                    <Search
                      className="absolute left-3 top-3 text-slate-500"
                      size={16}
                    />
                    <input
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Search sponsors..."
                      className="w-full rounded-xl border border-white/10 bg-[#11121b] py-2.5 pl-9 pr-3 text-sm outline-none focus:border-purple-400/50"
                    />
                  </div>
                </div>

                {errors.sponsors ? (
                  <ErrorText error={errors.sponsors} />
                ) : (
                  <Panel
                    title={`${filteredSponsors.length} sponsor profiles`}
                    subtitle="Returned by GET /api/v1/sponsors"
                  >
                    {sponsorLookupLoading ? (
                      <div className="py-10 text-center text-sm text-purple-300">
                        Searching sponsor profiles...
                      </div>
                    ) : filteredSponsors.length === 0 ? (
                      <Empty
                        text={
                          sponsorLookupError ||
                          "No matching sponsor profiles returned by the API."
                        }
                      />
                    ) : (
                      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                        {filteredSponsors.map((item, i) => {
                          const name =
                            item.sponsor ||
                            item.sponsor_name ||
                            item.name ||
                            `Sponsor ${i + 1}`;

                          return (
                            <button
                              key={item.sponsor_id || name}
                              onClick={() => openSponsor(item)}
                              className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-4 text-left transition hover:border-purple-400/30 hover:bg-purple-500/[0.04]"
                            >
                              <div className="flex items-start justify-between gap-3">
                                <div className="rounded-lg bg-purple-500/10 p-2 text-purple-300">
                                  <Users size={17} />
                                </div>
                                <ArrowUpRight size={15} className="text-slate-600" />
                              </div>

                              <p className="mt-4 line-clamp-2 text-sm font-medium text-slate-200">
                                {name}
                              </p>

                              <pre className="mt-2 line-clamp-3 whitespace-pre-wrap text-[11px] leading-5 text-slate-500">
                                {Object.entries(item)
                                  .filter(
                                    ([key]) =>
                                      !["sponsor", "sponsor_name", "name"].includes(key)
                                  )
                                  .slice(0, 3)
                                  .map(
                                    ([key, value]) =>
                                      `${key}: ${pretty(value)}`
                                  )
                                  .join("\n")}
                              </pre>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </Panel>
                )}
              </>
            )}

            {section === "Agent Memory" && (
              <>
                <div>
                  <h1 className="text-2xl font-semibold text-white">
                    Agent memory
                  </h1>
                  <p className="mt-2 text-sm text-slate-500">
                    Inspect recorded episodes and current procedural rules.
                  </p>
                </div>

                <Panel
                  title="Procedural memory"
                  subtitle="Current reasoning rules for a selected specialist agent"
                >
                  <div className="flex flex-wrap items-center gap-3">
                    <div className="relative min-w-0 flex-1">
                      <select
                        value={selectedAgent}
                        onChange={(e) => setSelectedAgent(e.target.value)}
                        className="w-full appearance-none rounded-xl border border-white/10 bg-[#0b0c13] px-4 py-3 pr-10 text-sm capitalize text-slate-200 outline-none focus:border-purple-400/50"
                      >
                        {AGENTS.map((agent) => (
                          <option key={agent} value={agent}>
                            {shortAgent(agent)}
                          </option>
                        ))}
                      </select>
                      <ChevronDown
                        className="pointer-events-none absolute right-3 top-3.5 text-slate-500"
                        size={16}
                      />
                    </div>

                    <button
                      onClick={() => loadProcedures(selectedAgent)}
                      className="rounded-xl border border-white/10 px-4 py-3 text-xs text-slate-300 hover:bg-white/5"
                    >
                      Reload rules
                    </button>
                  </div>

                  {errors.procedures && (
                    <div className="mt-4">
                      <ErrorText error={errors.procedures} />
                    </div>
                  )}

                  {proceduresData && (
                    <div className="mt-4 grid gap-3 sm:grid-cols-3">
                      <Metric
                        label="Total rules"
                        value={proceduresData.total_rules ?? "—"}
                        icon={BookOpen}
                      />
                      <Metric
                        label="Default rules"
                        value={proceduresData.default_rules ?? "—"}
                        icon={Database}
                      />
                      <Metric
                        label="Learned rules"
                        value={proceduresData.learned_rules ?? "—"}
                        icon={Brain}
                      />
                    </div>
                  )}

                  {proceduresData && (
                    <div className="mt-4 space-y-3">
                      {asArray(proceduresData, "procedures").length === 0 ? (
                        <Empty text="No procedural rules returned." />
                      ) : (
                        asArray(proceduresData, "procedures").map((rule, i) => (
                          <div
                            key={rule.id || i}
                            className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-4"
                          >
                            <div className="mb-2 flex items-center justify-between gap-3">
                              <p className="text-sm font-medium text-slate-200">
                                Rule {i + 1}
                              </p>
                              <StatusPill value={rule.rule_type || "default"} />
                            </div>

                            <p className="whitespace-pre-wrap text-sm leading-6 text-slate-400">
                              {rule.rule ||
                                rule.content ||
                                rule.instruction ||
                                pretty(rule)}
                            </p>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </Panel>

                <Panel
                  title="Episodic memory"
                  subtitle="Recent agent reasoning sessions"
                >
                  {errors.episodes ? (
                    <ErrorText error={errors.episodes} />
                  ) : episodes.length === 0 ? (
                    <Empty text="No memory episodes returned by the API." />
                  ) : (
                    <div className="space-y-3">
                      {episodes.map((episode, i) => (
                        <div
                          key={episode.episode_id || i}
                          className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-4"
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <p className="text-sm font-medium capitalize text-slate-200">
                              {shortAgent(episode.agent || "Agent")}
                            </p>
                            <span className="text-xs text-slate-600">
                              {formatDate(
                                episode.created_at || episode.timestamp
                              )}
                            </span>
                          </div>

                          <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-400">
                            {episode.outcome ||
                              episode.summary ||
                              episode.content ||
                              pretty(episode)}
                          </p>

                          {episode.nct_id && (
                            <p className="mt-2 font-mono text-xs text-purple-300">
                              {episode.nct_id}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </Panel>
              </>
            )}

            <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.06] pt-5 text-[11px] text-slate-600">
              <span>MOSAIC · Multi-Agent Operating System for AI Cognition</span>
              <span>
                Candidate signals require appropriate human interpretation.
              </span>
            </footer>
          </div>
        </div>
      </div>

      {(selectedSignal || selectedSponsor) && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
          onClick={() => {
            setSelectedSignal(null);
            setSelectedSponsor(null);
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="max-h-[85vh] w-full max-w-2xl overflow-auto rounded-2xl border border-white/10 bg-[#12131d] p-6 shadow-2xl"
          >
            <div className="mb-5 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">
                {selectedSignal ? "Signal details" : "Sponsor profile"}
              </h2>
              <button
                onClick={() => {
                  setSelectedSignal(null);
                  setSelectedSponsor(null);
                }}
                className="rounded-lg p-2 text-slate-400 hover:bg-white/5"
              >
                <X size={18} />
              </button>
            </div>

            <pre className="whitespace-pre-wrap break-words rounded-xl bg-black/30 p-4 text-xs leading-6 text-slate-300">
              {JSON.stringify(selectedSignal || selectedSponsor, null, 2)}
            </pre>
          </motion.div>
        </div>
      )}
    </main>
  );
}