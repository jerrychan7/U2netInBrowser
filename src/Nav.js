
import "./Nav.css";

export default function Nav() {
  return (
    <header className="header">
      <nav>
        <ul>
          <li><a href="#section-about">about</a></li>
          <li className="logo-text"><a href="#"><h1>salient</h1></a></li>
          <li><a href="#section-function">result</a></li>
        </ul>
      </nav>
    </header>
  );
};
