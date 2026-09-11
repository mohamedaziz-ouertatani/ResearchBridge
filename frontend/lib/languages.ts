/*
  Flag emoji per language code, for the corpus-stats "papers by language"
  chart. Flags represent countries, not languages, so this is the usual
  imprecise-but-legible convention (a flag stands in for "a place this
  language is associated with," not a claim of exclusivity) - skipped
  entirely for a handful of codes with no obvious single flag (ca, br) or
  no real flag emoji (br is a regional French language). Full names come
  from Intl.DisplayNames at render time rather than a second hardcoded
  table, so any code missing from this map still gets a readable name.

  Two special, deliberately distinct buckets - see routes.py's stats
  endpoint and core.py's connector docstring:
  - "unknown": Paper.language was NULL - the source never reported a
    language for this record at all (almost entirely CORE and Semantic
    Scholar, whose APIs don't reliably expose one).
  - "zz": ISO 639's own reserved code for "language undetermined" - the
    source DID report a language, and what it reported was "we don't
    know" (CORE-only, live-confirmed). Collapsing this into "unknown"
    would lose that distinction for no benefit.
*/
const LANGUAGE_FLAGS: Record<string, string> = {
  en: "🇬🇧",
  pt: "🇵🇹",
  es: "🇪🇸",
  uk: "🇺🇦",
  de: "🇩🇪",
  id: "🇮🇩",
  tr: "🇹🇷",
  sl: "🇸🇮",
  it: "🇮🇹",
  ja: "🇯🇵",
  ar: "🇸🇦",
  ru: "🇷🇺",
  pl: "🇵🇱",
  fi: "🇫🇮",
  hu: "🇭🇺",
  nl: "🇳🇱",
  vi: "🇻🇳",
  fr: "🇫🇷",
  cs: "🇨🇿",
  hr: "🇭🇷",
  ka: "🇬🇪",
  yo: "🇳🇬",
  nb: "🇳🇴",
  ko: "🇰🇷",
  sq: "🇦🇱",
  fa: "🇮🇷",
  sk: "🇸🇰",
  ro: "🇷🇴",
};

let displayNames: Intl.DisplayNames | null = null;
try {
  displayNames = new Intl.DisplayNames(["en"], { type: "language" });
} catch {
  displayNames = null;
}

/** "en" -> "🇬🇧 English". Falls back to the raw code if the runtime can't
 * resolve a display name for it. The two non-ISO-639-1 sentinel codes
 * above ("unknown", "zz") are handled by name, not looked up. */
export function formatLanguage(code: string): string {
  if (code === "unknown") return "not reported";
  if (code === "zz") return "undetermined (source-reported)";

  const flag = LANGUAGE_FLAGS[code];
  let name: string;
  try {
    name = displayNames?.of(code) ?? code;
  } catch {
    name = code;
  }
  return flag ? `${flag} ${name}` : name;
}
