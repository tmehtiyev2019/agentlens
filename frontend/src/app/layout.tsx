import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgentLens — Moonshot TEA Evaluator",
  description:
    "Multi-agent system for evaluating early-stage moonshot ideas using LangGraph, Claude, and RAGAS.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0F172A] text-white antialiased">
        {children}
      </body>
    </html>
  );
}
