import { AssessmentComparison } from "@/components/AssessmentComparison";

type ComparePageProps = {
  searchParams: Promise<{ ids?: string | string[] }>;
};

export default async function ComparePage({ searchParams }: ComparePageProps) {
  const params = await searchParams;
  const rawIds = params.ids;
  const ids = (typeof rawIds === "string" ? rawIds.split(",") : [])
    .filter(Boolean)
    .slice(0, 4);

  return <AssessmentComparison ids={ids} />;
}
