import { useRef, useState } from "react";
import "./SignIn.css";

const SignIn = ({ setIsPro, setPasscode, setToken }) => {
  const [isVisible, setVisible] = useState(0);
  const inputRef = useRef(null);

  const toggleVisibility = () => {
    setVisible(1 - isVisible);
  };

  const handleChange = (e) => {
    // Remove error message
    document.querySelectorAll(".form-control").forEach((element) => {
      element.classList.remove("is-invalid");
    });
    document.querySelectorAll(".form-help").forEach((element) => {
      element.classList.remove("text-danger");
    });
    document.querySelectorAll(".show-password-button").forEach((element) => {
      element.classList.remove("show-password-button-on-error");
    });

    // If passcode is not empty, enable the submit button
    if (e.target.value !== "") {
      document.getElementById("sign-in-button").disabled = false;
    } else {
      document.getElementById("sign-in-button").disabled = true;
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const passcode = inputRef.current.value;

    // // POST request to server
    console.log("inside");
    // fetch("http://localhost:8000/auth/signin", {
      fetch(window.API_URL + "/auth/signin", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ passcode: passcode }),
    }).then((response) => {
      // If response is ok, set passcode
      console.log(response);
      if (response.ok) {
        console.log("@ok");
        // Get token from response
        response.json().then((data) => {
          // Set token

          localStorage.setItem("token", data.token);
          setToken(data.token);
          setIsPro(data.is_pro);

          console.log(data);
          setPasscode(passcode);
        });
      } else {
        console.log("@else");
        // Else, show error message
        document.getElementById("floatingPassword").classList.add("is-invalid");
        document.getElementById("password-label").classList.add("text-danger");
        document.getElementById("show-password").classList.add("show-password-button-on-error");
        // Disable submit button
        document.getElementById("sign-in-button").disabled = true;
      }
    });
  };

  return (
    <div id="form-frame">
      <h1 className="form-title">Sign In</h1>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="password" className="form-help" id="password-label">
            Passcode
          </label>
          <input
            type={isVisible ? "text" : "password"}
            ref={inputRef}
            maxLength="30"
            onChange={handleChange}
            className="form-control sign-in-input"
            id="floatingPassword"
          />
          <label className="invalid-feedback">Invalid passcode</label>
          <button id="show-password" className="show-password-button" type="button" onClick={toggleVisibility}>
            <span className={isVisible ? "bi-eye-slash" : "bi-eye"} />
          </button>
        </div>
        <button type="submit" id="sign-in-button" className="submit-button sign-in-submit-button" disabled>
          SIGN IN
        </button>
      </form>
    </div>
  );
};

export default SignIn;
