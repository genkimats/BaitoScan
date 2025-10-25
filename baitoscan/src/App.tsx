import Home from "./pages/HomePage";
import Settings from "./pages/SettingsPage";

function App() {
  const path = window.location.pathname;
  if (path.startsWith("/settings")) return <Settings />;
  return <Home />;
}

export default App;
