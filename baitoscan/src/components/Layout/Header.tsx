export default function Header() {
  return (
    <header className="bg-blue-500 text-white px-4 py-3 flex justify-between items-center shadow">
      <h1 className="font-bold text-lg">BaitoScan</h1>
      <nav className="text-sm flex gap-4">
        <a href="/" className="hover:underline">Home</a>
        <a href="/settings" className="hover:underline">Settings</a>
      </nav>
    </header>
  );
}
