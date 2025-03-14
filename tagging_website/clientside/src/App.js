import { useState } from "react";
import MainView from "./components/classification/MainView";
import SignIn from "./components/auth/SignIn";
import MainViewRefactored from "./components/classification/MainViewRefactored";

const App = () => {
  // Current signed in user
  const [passcode, setPasscode] = useState("");
  const [token, setToken] = useState(null);
  const [isPro, setIsPro] = useState(true);

  return (
    <div className="App">
      <main className={passcode ? "" : "plaster"}>
        {
          // Check if user is signed in
          passcode ? (
            // <MainView passcode={passcode} isPro={isPro} setPasscode={setPasscode} token={token} setToken={setToken} />
              <MainViewRefactored token={token} />
          ) : (
            <SignIn setIsPro={setIsPro} setPasscode={setPasscode} setToken={setToken} />
          )
        }
      </main>
    </div>
  );
};

export default App;
