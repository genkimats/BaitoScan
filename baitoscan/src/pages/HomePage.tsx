import PageWrapper from "../components/Layout/PageWrapper";

export default function Home() {
  return (
    <PageWrapper>
      <div className="text-center grid gap-4">
        <h2 className="text-2xl font-semibold">Welcome to BaitoScan 👋</h2>
        <p className="text-gray-600">
          Snap a photo of your handwritten shift sheet and instantly calculate your salary.
        </p>
      </div>
    </PageWrapper>
  );
}
