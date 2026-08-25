import type { Metadata } from "next";
import { Space_Grotesk, Source_Serif_4, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { NotificationBell } from "@/components/NotificationBell";

/*
  Three faces, each tied to a kind of content:
    Space Grotesk  - instrument chrome (labels, headings, controls)
    Source Serif 4 - the literature itself (titles, abstracts)
    JetBrains Mono - identifiers and measurements (arXiv ids, distances)
*/
// Names are deliberately ff-* rather than font-*: Tailwind v4 ships its own
// --font-sans/--font-serif/--font-mono theme keys, and a same-named variable
// here loses to them, silently falling every face back to a system default.
const grotesk = Space_Grotesk({
  variable: "--ff-grotesk",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const serif = Source_Serif_4({
  variable: "--ff-serif",
  subsets: ["latin"],
  weight: ["400", "600"],
  style: ["normal", "italic"],
});

const mono = JetBrains_Mono({
  variable: "--ff-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "ResearchBridge — research assessment",
  description:
    "Check a research idea or paper against the literature: what is already solved, what gap remains, and what the evidence actually supports.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // font vars go on <html>, not <body>: globals.css declares --type-* at
    // :root, and a custom property resolves var() where it is declared - so
    // --ff-* has to exist at that same level or the declaration is invalid.
    <html lang="en" className={`${grotesk.variable} ${serif.variable} ${mono.variable}`}>
      <body>
        {children}
        <NotificationBell />
      </body>
    </html>
  );
}
