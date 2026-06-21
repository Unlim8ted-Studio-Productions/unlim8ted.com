(function () {
  function createAvatarMarkup(options) {
    const isAstronaut = options?.variant === "astronaut";
    const variant = isAstronaut ? "meatballAstronaut" : "meatballAstronaut meatballAstronaut--plain";
    const label = options?.label || "Meatball";
    const shellMarkup = isAstronaut
      ? `
        <div class="meatballAstronaut-suit" aria-hidden="true">
          <div class="meatballAstronaut-suitTorso"></div>
          <div class="meatballAstronaut-suitArm meatballAstronaut-suitArm-left"></div>
          <div class="meatballAstronaut-suitArm meatballAstronaut-suitArm-right"></div>
          <div class="meatballAstronaut-suitLeg meatballAstronaut-suitLeg-left"></div>
          <div class="meatballAstronaut-suitLeg meatballAstronaut-suitLeg-right"></div>
          <div class="meatballAstronaut-boot meatballAstronaut-boot-left"></div>
          <div class="meatballAstronaut-boot meatballAstronaut-boot-right"></div>
          <div class="meatballAstronaut-glove meatballAstronaut-glove-left"></div>
          <div class="meatballAstronaut-glove meatballAstronaut-glove-right"></div>
          <div class="meatballAstronaut-pack"></div>
        </div>
        <div class="meatballAstronaut-jet meatballAstronaut-jet-left" aria-hidden="true"></div>
        <div class="meatballAstronaut-jet meatballAstronaut-jet-right" aria-hidden="true"></div>
        <div class="meatballAstronaut-helmetShell" aria-hidden="true"></div>
        <div class="meatballAstronaut-helmet" aria-hidden="true"></div>
        <div class="meatballAstronaut-helmetRim" aria-hidden="true"></div>
        <div class="meatballAstronaut-tube" aria-hidden="true"></div>
        <div class="meatballAstronaut-collar" aria-hidden="true"></div>
      `
      : "";
    return `
      <div class="${variant}" data-meatball-variant="${options?.variant || "plain"}">
        ${shellMarkup}
        <div id="${options?.avatarId || "bigMeatballAvatar"}" class="meatballAvatar" aria-label="${label}">
          <span class="meatballAvatar-shadow"></span>
          <span class="meatballAvatar-body">
            <span class="meatballAvatar-eye left"><span class="pupil"></span></span>
            <span class="meatballAvatar-eye right"><span class="pupil"></span></span>
            <span class="meatballAvatar-mouth"></span>
          </span>
        </div>
      </div>
    `;
  }

  function resolveAvatar(target) {
    if (!target) return null;
    if (target.classList?.contains("meatballAvatar")) return target;
    return target.querySelector?.(".meatballAvatar") || null;
  }

  function setTalking(target, talking) {
    const avatar = resolveAvatar(target);
    if (!avatar) return null;
    avatar.classList.toggle("talking", Boolean(talking));
    return avatar;
  }

  function speakFor(target, text, maxMs) {
    const avatar = setTalking(target, true);
    if (!avatar) return 0;
    const duration = Math.min(
      typeof maxMs === "number" ? maxMs : 4200,
      Math.max(900, String(text || "").length * 34)
    );
    window.setTimeout(() => {
      avatar.classList.remove("talking");
    }, duration);
    return duration;
  }

  function mount(target, options) {
    const container = typeof target === "string" ? document.querySelector(target) : target;
    if (!container) return null;
    container.innerHTML = createAvatarMarkup(options);
    return resolveAvatar(container);
  }

  window.MeatballAvatarEmbed = {
    mount,
    setTalking,
    speakFor,
    resolveAvatar
  };
})();
