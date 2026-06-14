const state = {
  selectedProject: window.localStorage.getItem("dancinghippo.selectedProject") || "",
  status: null,
  project: null,
  projectListSignature: "",
  projectViewSignature: "",
  languageProjectName: "",
  briefProjectName: "",
  pendingRegenerationSceneNumber: null,
  pollTimer: null,
  manualNewProject: false,
};

const $ = (id) => document.getElementById(id);
const SELECTED_PROJECT_KEY = "dancinghippo.selectedProject";

function rememberSelectedProject(projectName) {
  state.selectedProject = projectName;
  if (projectName) {
    window.localStorage.setItem(SELECTED_PROJECT_KEY, projectName);
  } else {
    window.localStorage.removeItem(SELECTED_PROJECT_KEY);
  }
}

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function splitSize(value) {
  const [width, height] = value.split("x").map((part) => Number.parseInt(part, 10));
  return { width, height };
}

function escapeAttr(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("\"", "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function jobPayload() {
  const keyframeSize = splitSize($("keyframeSize").value);
  const videoSize = splitSize($("videoSize").value);
  return {
    project_name: $("projectName").value.trim(),
    brief: $("briefText").value.trim(),
    is_script: false,
    lazy: $("lazyMode").checked,
    takes_per_scene: Number.parseInt($("takesPerScene").value, 10),
    scene_min: Number.parseInt($("sceneMin").value, 10),
    scene_max: Number.parseInt($("sceneMax").value, 10),
    use_keyframes: $("useKeyframes").checked,
    skip_keyframe_eval: $("skipKeyframeEval").checked,
    keyframe_width: keyframeSize.width,
    keyframe_height: keyframeSize.height,
    video_width: videoSize.width,
    video_height: videoSize.height,
    output_language: $("outputLanguage").value,
  };
}

function setMessage(text) {
  const message = $("actionMessage");
  message.textContent = text;
  message.classList.toggle("active", Boolean(text));
}

function activeJobForProject(status, project) {
  if (!status.job?.project_name || status.job.project_name !== project?.name) {
    return null;
  }
  return status.job;
}

function projectHasFinal(project) {
  return Boolean(project?.state?.final_file?.exists);
}

function isProjectComplete(project) {
  return project?.phase === "complete" || projectHasFinal(project);
}

function isProjectLockedForStoryboardEdits(project) {
  return isProjectComplete(project) || ["review_takes", "generating_video"].includes(project?.phase || "");
}

function refreshProjectName(status) {
  const typedProjectName = $("projectName").value.trim();
  if (typedProjectName) {
    return typedProjectName;
  }
  if (state.manualNewProject) {
    return "";
  }
  if (state.selectedProject) {
    return state.selectedProject;
  }
  return status.projects?.[0]?.name || "";
}

function activeRegenerationSceneNumber(status, project) {
  const job = activeJobForProject(status, project);
  if (
    job
    && ["queued", "running"].includes(job.status)
    && job.action?.type === "regenerate_keyframe"
  ) {
    return Number.parseInt(job.action.scene_number, 10);
  }
  return state.pendingRegenerationSceneNumber;
}

function regenerationStateForScene(status, project, scene) {
  const sceneNumber = Number.parseInt(scene.scene_number, 10);
  const job = activeJobForProject(status, project);
  const action = job?.action || null;
  const actionSceneNumber = Number.parseInt(action?.scene_number, 10);
  if (action?.type === "regenerate_keyframe" && actionSceneNumber === sceneNumber) {
    if (["queued", "running"].includes(job.status)) {
      return { status: "running", label: "Regenerating...", disabled: true };
    }
    if (job.status === "error") {
      return { status: "error", label: "Retry Keyframe", disabled: false };
    }
    if (job.status === "finished") {
      return { status: "completed", label: "Regenerated", disabled: true };
    }
  }
  if (state.pendingRegenerationSceneNumber === sceneNumber) {
    return { status: "running", label: "Regenerating...", disabled: true };
  }
  if (scene.keyframe_regeneration_requested_at) {
    return { status: "completed", label: "Regenerated", disabled: true };
  }
  return { status: "idle", label: "Regenerate Keyframe", disabled: false };
}

function phaseLabel(phase) {
  const labels = {
    planning: "Planning",
    generating_storyboard: "Generating Storyboard",
    review_storyboard: "Review Storyboard",
    ready_for_video: "Ready for Video",
    generating_video: "Generating Video",
    review_takes: "Review Takes",
    complete: "Complete",
    error: "Error",
  };
  return labels[phase] || "Idle";
}

function setActiveStep(phase) {
  const stepMap = {
    planning: "planning",
    generating_storyboard: "storyboard",
    review_storyboard: "storyboard",
    ready_for_video: "video",
    generating_video: "video",
    review_takes: "video",
    complete: "final",
    error: "storyboard",
  };
  const active = stepMap[phase] || "planning";
  document.querySelectorAll(".flow-step").forEach((node) => {
    node.classList.toggle("active", node.dataset.step === active);
  });
}

function renderHealth(status) {
  const health = status.health;
  $("comfyStatus").textContent = `ComfyUI ${health.comfyui.ok ? "Online" : "Offline"}`;
  $("comfyStatus").className = `health-item ${health.comfyui.ok ? "ok" : "bad"}`;
  $("ollamaStatus").textContent = `Ollama ${health.ollama.ok ? "Online" : "Offline"}`;
  $("ollamaStatus").className = `health-item ${health.ollama.ok ? "ok" : "bad"}`;
  $("modelStatus").textContent = health.ollama.model;
  $("modelStatus").className = "health-item ok";
}

function renderProjectList(projects) {
  const activeProjectName = state.project?.exists ? state.project.name : "";
  const signature = JSON.stringify(projects.map((project) => [
    project.name,
    project.phase,
    project.scene_count,
    project.keyframe_count,
    project.take_count,
    project.reference_count,
    project.final_exists,
    project.updated_at,
  ]).concat([["active", activeProjectName]]));
  if (signature === state.projectListSignature) {
    return;
  }
  state.projectListSignature = signature;
  const list = $("projectList");
  list.innerHTML = "";
  $("newProjectButton").classList.toggle("active", !activeProjectName);
  if (!projects.length) {
    list.innerHTML = `<div class="empty-state">No projects yet.</div>`;
    return;
  }
  projects.forEach((project) => {
    const button = document.createElement("button");
    button.className = `project-button ${project.name === activeProjectName ? "active" : ""}`;
    button.type = "button";
    button.innerHTML = `<strong>${project.name}</strong><br><span>${phaseLabel(project.phase)}</span>`;
    button.addEventListener("click", () => {
      state.manualNewProject = false;
      rememberSelectedProject(project.name);
      $("projectName").value = project.name;
      refresh();
    });
    list.appendChild(button);
  });
}

function renderSummary(project) {
  const stateData = project?.state;
  if (!stateData) {
    $("currentProjectSummary").innerHTML = `<div class="empty-state">Start a production to see live status.</div>`;
    return;
  }
  const scenes = stateData.scenes || [];
  const keyframes = scenes.filter((scene) => scene.keyframe_candidates?.length).length;
  const takes = scenes.filter((scene) => scene.takes?.length).length;
  const references = stateData.project_references?.length || 0;
  $("currentProjectSummary").innerHTML = `
    <div class="metric"><strong>${scenes.length}</strong><span>Scenes planned</span></div>
    <div class="metric"><strong>${keyframes}/${scenes.length}</strong><span>Keyframes ready</span></div>
    <div class="metric"><strong>${takes}/${scenes.length}</strong><span>Video takes ready</span></div>
    <div class="metric"><strong>${references}</strong><span>Project references</span></div>
  `;
}

function renderProjectReferences(project) {
  const references = project?.state?.project_references || [];
  const locked = isProjectLockedForStoryboardEdits(project);
  $("uploadReferenceButton").disabled = locked;
  $("referenceKind").disabled = locked;
  $("referenceLabel").disabled = locked;
  $("referenceFile").disabled = locked;
  $("projectReferenceStatus").textContent = references.length
    ? `${references.length} reference${references.length === 1 ? "" : "s"} saved.${locked ? " Locked after storyboard." : ""}`
    : "No references added.";
  if (!references.length) {
    $("projectReferenceList").innerHTML = `<div class="reference-empty">${locked ? "References are locked after storyboard generation." : "Add character, style, location, or prop images before generating keyframes."}</div>`;
    return;
  }
  $("projectReferenceList").innerHTML = references.map((reference) => {
    const image = reference.file?.exists
      ? `<img src="${reference.file.url}" alt="${escapeAttr(reference.label)} reference">`
      : `<div class="reference-missing">Missing</div>`;
    return `
      <div class="project-reference-item">
        ${image}
        <div>
          <strong>${escapeHtml(reference.label)}</strong>
          <small>${escapeHtml(reference.kind)}${reference.original_name ? ` · ${escapeHtml(reference.original_name)}` : ""}</small>
        </div>
        <button class="mini-button project-reference-delete" type="button" data-reference-id="${escapeAttr(reference.id)}" ${locked ? "disabled" : ""}>Delete</button>
      </div>
    `;
  }).join("");
}

function renderBible(project) {
  const stateData = project?.state;
  const grid = $("bibleGrid");
  if (!stateData) {
    $("bibleStatus").textContent = "No project loaded";
    grid.innerHTML = `<div class="empty-state">Start or load a project to see the production bible.</div>`;
    return;
  }
  const characters = stateData.characters || {};
  const voices = stateData.voices || {};
  const references = stateData.character_references || {};
  const style = stateData.style || "No style lock stored yet.";
  const directorNotes = stateData.director_notes || {};
  const scenes = stateData.scenes || [];
  const locked = isProjectLockedForStoryboardEdits(project);
  const settings = [...new Set(scenes.map((scene) => scene.setting_description).filter(Boolean))];
  $("bibleStatus").textContent = `${Object.keys(characters).length} characters · ${settings.length} locations`;
  const characterItems = Object.entries(characters).map(([id, description]) => {
    const reference = references[id];
    const referenceBlock = reference?.file?.exists ? `
      <div class="reference-card">
        <img src="${reference.file.url}" alt="${id} reference">
        <div>
          <strong>Reference locked</strong>
          <small>Scene ${reference.source_scene_number || "?"} · Keyframe ${reference.source_candidate || "?"}</small>
          <button class="mini-button reference-clear" type="button" data-character-id="${escapeAttr(id)}" ${locked ? "disabled" : ""}>Clear</button>
        </div>
      </div>
    ` : `<small>No locked reference image.</small>`;
    return `
      <div class="bible-item">
        <strong>${id}</strong>
        <p>${description}</p>
        ${voices[id] ? `<small>Voice: ${voices[id]}</small>` : ""}
        ${referenceBlock}
      </div>
    `;
  }).join("");
  const settingItems = settings.slice(0, 4).map((setting) => `
    <div class="bible-item">
      <strong>Location</strong>
      <p>${setting}</p>
    </div>
  `).join("");
  const fixedItems = Array.isArray(directorNotes.problems_fixed)
    ? directorNotes.problems_fixed.map((item) => `<li>${item}</li>`).join("")
    : "";
  grid.innerHTML = `
    <div class="bible-column">
      <h4>Director Pass</h4>
      <div class="bible-item">
        <strong>${directorNotes.story_grade ? `Story grade: ${directorNotes.story_grade}` : "Not generated yet"}</strong>
        <p>${directorNotes.logline || "New projects will run a director rewrite pass before storyboard generation."}</p>
        ${directorNotes.pacing_notes ? `<small>Pacing: ${directorNotes.pacing_notes}</small>` : ""}
        ${directorNotes.continuity_strategy ? `<small>Continuity: ${directorNotes.continuity_strategy}</small>` : ""}
        ${fixedItems ? `<ul class="bible-list">${fixedItems}</ul>` : ""}
      </div>
      <h4>Character Bible</h4>
      ${characterItems || `<div class="empty-state">No characters stored yet.</div>`}
    </div>
    <div class="bible-column">
      <h4>Style & Scene Bible</h4>
      <div class="bible-item"><strong>Style lock</strong><p>${style}</p></div>
      ${settingItems || `<div class="empty-state">No reusable settings stored yet.</div>`}
    </div>
  `;
}

function renderLanguage(project, status) {
  const projectName = project?.name || "";
  if (state.languageProjectName === projectName) {
    return;
  }
  const savedLanguage = project?.state?.output_language || status.health?.output_language || "english";
  if ($("outputLanguage").value !== savedLanguage) {
    $("outputLanguage").value = savedLanguage;
  }
  state.languageProjectName = projectName;
}

function progressInfo(project, status) {
  const stateData = project?.state;
  if (!stateData) {
    return {
      title: "Waiting",
      detail: "Load a project to see production progress.",
      percent: 0,
    };
  }
  const scenes = stateData.scenes || [];
  const totalScenes = scenes.length;
  const keyframes = scenes.filter((scene) => scene.keyframe_candidates?.length).length;
  const approvedKeyframes = scenes.filter((scene) => scene.keyframe_approved || scene.selected_keyframe).length;
  const takes = scenes.filter((scene) => scene.takes?.length).length;
  const selectedTakes = scenes.filter((scene) => scene.selected_take).length;
  const phase = project?.phase || "planning";
  const job = activeJobForProject(status, project);
  const activeJobStatus = job?.status || null;
  const running = activeJobStatus === "running" || activeJobStatus === "queued";
  const latestLog = status.job?.logs?.length ? status.job.logs[status.job.logs.length - 1] : "";

  if (activeJobStatus === "error" || stateData.last_error) {
    const errorText = status.job?.error
      ? String(status.job.error).split("\n")[0]
      : stateData.last_error?.message
        ? String(stateData.last_error.message).split("\n")[0]
      : "Production stopped with an error. Check the log, then retry.";
    return {
      title: "Production Error",
      detail: errorText,
      percent: totalScenes ? Math.max(12, Math.round((keyframes / totalScenes) * 46)) : 0,
    };
  }
  if (running) {
    if (job?.action?.type === "regenerate_keyframe") {
      const sceneNumber = Number.parseInt(job.action.scene_number, 10);
      const detail = latestLog || `Scene ${sceneNumber} keyframe regeneration has started.`;
      return {
        title: `Regenerating Scene ${sceneNumber} Keyframe`,
        detail,
        percent: 46,
      };
    }
    const detail = latestLog || "Backend is running. The page will update automatically.";
    const percent = totalScenes ? Math.min(92, 18 + Math.round((takes / totalScenes) * 58)) : 12;
    return { title: "Running", detail, percent };
  }
  if (phase === "complete") {
    return {
      title: "Complete",
      detail: `Final film ready. ${selectedTakes}/${totalScenes} selected takes assembled.`,
      percent: 100,
    };
  }
  if (phase === "review_takes") {
    return {
      title: "Review Takes",
      detail: `${takes}/${totalScenes} scenes have video takes. Select the takes you want, then assemble.`,
      percent: 86,
    };
  }
  if (phase === "ready_for_video") {
    return {
      title: "Ready for Video",
      detail: `${approvedKeyframes}/${totalScenes} storyboard keyframes approved.`,
      percent: 62,
    };
  }
  if (phase === "review_storyboard") {
    return {
      title: "Review Storyboard",
      detail: `${keyframes}/${totalScenes} keyframes ready. Approve storyboard to continue.`,
      percent: 46,
    };
  }
  if (phase === "generating_storyboard") {
    return {
      title: "Generating Storyboard",
      detail: `${keyframes}/${totalScenes} keyframes ready so far.`,
      percent: totalScenes ? 20 + Math.round((keyframes / totalScenes) * 22) : 20,
    };
  }
  return {
    title: "Script Plan",
    detail: totalScenes ? `${totalScenes} scenes planned.` : "Scene planning has not run yet.",
    percent: totalScenes ? 22 : 0,
  };
}

function renderProgress(project, status) {
  const info = progressInfo(project, status);
  $("progressTitle").textContent = info.title;
  $("progressDetail").textContent = info.detail;
  $("progressFill").style.width = `${info.percent}%`;
}

function candidateScore(candidate, selectedPath) {
  if (!candidate?.file?.exists) {
    return 0;
  }
  const evalData = candidate.eval || {};
  if (candidate.path === selectedPath) {
    return 100;
  }
  if (evalData.verdict === "PASS" && evalData.quality_gate !== "warning") {
    return 80;
  }
  if (evalData.verdict === "PASS" && evalData.quality_gate === "warning") {
    return 70;
  }
  if (evalData.quality_gate === "manual_review" || evalData.notes === "Evaluation skipped") {
    return 60;
  }
  if (candidate.status === "generated") {
    return 40;
  }
  return 10;
}

function bestKeyframeCandidate(scene) {
  const candidates = scene.keyframe_candidates || [];
  const selectedPath = scene.selected_keyframe_file?.path || scene.selected_keyframe;
  return candidates
    .filter((item) => item.file?.exists)
    .sort((left, right) => candidateScore(right, selectedPath) - candidateScore(left, selectedPath))[0];
}

function candidateThumb(scene) {
  const candidate = bestKeyframeCandidate(scene);
  if (!candidate) {
    return `<div class="thumb">No keyframe</div>`;
  }
  return `<div class="thumb"><img src="${candidate.file.url}" alt="Scene ${scene.scene_number} keyframe"><span>Candidate #${candidate.candidate}</span></div>`;
}

function keyframeQuality(scene) {
  const candidates = scene.keyframe_candidates || [];
  if (!candidates.length) {
    return { label: "Keyframe waiting", tone: "waiting", detail: "No keyframe candidates generated yet." };
  }
  const selected = bestKeyframeCandidate(scene);
  const evalData = selected?.eval || {};
  if (evalData.quality_gate === "manual_review" || evalData.notes === "Evaluation skipped") {
    return {
      label: "Manual review",
      tone: "manual",
      detail: evalData.warning_reason || evalData.character_notes || "AI keyframe evaluation was skipped for this candidate.",
    };
  }
  if (evalData.verdict === "PASS" && evalData.quality_gate === "warning") {
    return {
      label: "AI WARN",
      tone: "manual",
      detail: evalData.warning_reason || evalData.character_notes || "Usable storyboard frame with minor mismatches.",
    };
  }
  if (evalData.verdict === "PASS") {
    return {
      label: "AI PASS",
      tone: "pass",
      detail: evalData.character_notes || "Character, setting, composition, and image quality passed.",
    };
  }
  if (evalData.verdict === "FAIL") {
    return {
      label: "AI FAIL",
      tone: "fail",
      detail: evalData.fail_reason || evalData.character_notes || "This keyframe failed quality review.",
    };
  }
  return {
    label: "Review needed",
    tone: "manual",
    detail: "No AI quality verdict is stored for this keyframe.",
  };
}

function qualityBreakdown(scene) {
  const candidates = scene.keyframe_candidates || [];
  if (!candidates.length) {
    return "";
  }
  const rows = candidates.map((candidate) => {
    const evalData = candidate.eval || {};
    const verdict = evalData.quality_gate === "manual_review" || evalData.notes === "Evaluation skipped"
      ? "MANUAL"
      : evalData.quality_gate === "warning"
        ? "WARN"
        : (evalData.verdict || "?");
    const tone = verdict === "PASS" ? "pass" : verdict === "FAIL" ? "fail" : "manual";
    const reason = evalData.fail_reason || evalData.warning_reason || evalData.character_notes || evalData.notes || "No notes";
    const recommended = bestKeyframeCandidate(scene)?.candidate === candidate.candidate
      ? `<em>Recommended</em>`
      : "";
    return `<div class="quality-row ${tone}"><strong>#${candidate.candidate} ${verdict}</strong><span>${reason}</span>${recommended}</div>`;
  }).join("");
  return `<div class="quality-breakdown">${rows}</div>`;
}

function firstGeneratedKeyframe(scene) {
  return bestKeyframeCandidate(scene);
}

function referenceControls(scene, references, locked) {
  const candidate = firstGeneratedKeyframe(scene);
  const characters = scene.characters_in_scene || [];
  if (locked || !candidate || !characters.length) {
    return "";
  }
  const buttons = characters.map((characterId) => {
    const locked = Boolean(references[characterId]?.file?.exists);
    return `
      <button
        class="reference-option ${locked ? "locked" : ""}"
        type="button"
        data-character-id="${escapeAttr(characterId)}"
        data-image-path="${escapeAttr(candidate.path)}"
        data-scene-number="${scene.scene_number}"
        data-candidate-number="${candidate.candidate}"
      >
        ${locked ? "Update" : "Lock"} ${characterId} reference
      </button>
    `;
  }).join("");
  return `<div class="reference-strip">${buttons}</div>`;
}

function takeThumb(scene) {
  const selectedPath = scene.selected_take_file?.path || scene.selected_take;
  const selected = scene.takes?.find((item) => item.file?.exists && item.path === selectedPath);
  const take = selected || scene.takes?.find((item) => item.file?.exists);
  if (!take) {
    return "";
  }
  return `<div class="thumb"><video src="${take.file.url}" muted controls></video></div>`;
}

function takeControls(scene) {
  const takes = scene.takes || [];
  if (!takes.length) {
    return "";
  }
  const selectedPath = scene.selected_take_file?.path || scene.selected_take;
  const items = takes.map((take) => {
    const isSelected = selectedPath === take.path;
    const disabled = take.status !== "generated" || !take.file?.exists;
    return `
      <button
        class="take-option ${isSelected ? "selected" : ""}"
        type="button"
        data-scene-number="${scene.scene_number}"
        data-take-number="${take.take}"
        ${disabled ? "disabled" : ""}
      >
        Take ${take.take}${isSelected ? " · Selected" : ""}
      </button>
    `;
  }).join("");
  return `<div class="take-strip">${items}</div>`;
}

function renderScenes(project, status) {
  const stateData = project?.state;
  const list = $("sceneList");
  if (!stateData?.scenes?.length) {
    $("sceneCountLabel").textContent = "No scenes yet";
    list.innerHTML = `<div class="empty-state">Scene planning has not run yet.</div>`;
    return;
  }
  const scenes = stateData.scenes;
  const references = stateData.character_references || {};
  const job = activeJobForProject(status, project);
  const projectJobRunning = Boolean(job && ["queued", "running"].includes(job.status));
  const regeneratingSceneNumber = activeRegenerationSceneNumber(status, project);
  const regenerationRunning = Boolean(regeneratingSceneNumber);
  const locked = isProjectLockedForStoryboardEdits(project);
  $("sceneCountLabel").textContent = `${scenes.length} scenes`;
  list.innerHTML = scenes.map((scene) => {
    const hasKeyframe = Boolean(scene.keyframe_candidates?.length);
    const hasTake = Boolean(scene.takes?.length);
    const keyframeDot = hasKeyframe ? "done" : "waiting";
    const takeDot = hasTake ? "done" : "waiting";
    const quality = keyframeQuality(scene);
    const isRegenerating = Number.parseInt(scene.scene_number, 10) === regeneratingSceneNumber;
    const regenerationState = regenerationStateForScene(status, project, scene);
    const actionStatus = {
      running: "Keyframe regeneration is running.",
      completed: "Keyframe regenerated.",
      error: "Regeneration failed. Try again.",
    }[regenerationState.status] || "";
    const disableRegenerate = locked || projectJobRunning || regenerationRunning || regenerationState.disabled;
    const regenerateLabel = locked && regenerationState.status === "idle"
      ? "Locked"
      : regenerationState.label;
    return `
      <article class="scene-row ${isRegenerating ? "working" : ""} ${regenerationState.status === "completed" ? "regenerated" : ""}">
        <div>
          ${candidateThumb(scene)}
          ${takeThumb(scene)}
        </div>
        <div class="scene-copy">
          <h4>Scene ${scene.scene_number} · ${scene.shot_type || "shot"} · ${scene.duration_seconds || "?"}s</h4>
          <div class="quality-pill ${quality.tone}">${quality.label}</div>
          <p>${scene.description || ""}</p>
          ${scene.dialogue ? `<p><strong>Dialogue:</strong> ${scene.dialogue}</p>` : ""}
          ${referenceControls(scene, references, locked)}
          ${takeControls(scene)}
          <div class="scene-actions">
            <button
              class="secondary-button scene-regenerate"
              type="button"
              data-scene-number="${scene.scene_number}"
              ${disableRegenerate ? "disabled" : ""}
            >${regenerateLabel}</button>
            ${actionStatus ? `<span class="scene-action-status ${regenerationState.status}">${actionStatus}</span>` : ""}
          </div>
          ${qualityBreakdown(scene)}
        </div>
        <div class="scene-meta">
          <div><span class="status-dot ${keyframeDot}"></span>Keyframe ${hasKeyframe ? "ready" : "waiting"}</div>
          <div><span class="status-dot ${quality.tone === "pass" ? "done" : quality.tone === "fail" ? "fail" : "waiting"}"></span>${quality.detail}</div>
          <div><span class="status-dot ${takeDot}"></span>Take ${hasTake ? "ready" : "waiting"}</div>
          <div>Mood: ${scene.mood || "not set"}</div>
          <div>Characters: ${(scene.characters_in_scene || []).join(", ") || "none"}</div>
        </div>
      </article>
    `;
  }).join("");
}

function renderFinal(project) {
  const final = project?.state?.final_file;
  const panel = document.querySelector(".final-panel");
  panel.hidden = !final?.exists && !["review_takes", "complete"].includes(project?.phase || "");
  if (!final?.exists) {
    $("finalStatus").textContent = "Not assembled";
    $("finalFilm").innerHTML = `<div class="empty-state">Assemble a final film after video takes are ready.</div>`;
    return;
  }
  $("finalStatus").textContent = "Ready";
  $("finalFilm").innerHTML = `<video src="${final.url}" controls></video>`;
}

function renderLogs(status, project) {
  const job = activeJobForProject(status, project);
  if (!job) {
    const idleText = isProjectComplete(project)
      ? "No active job. Project complete."
      : project?.exists
        ? "No active job. Project loaded."
        : "No active job. Start production or load a project.";
    if ($("logBox").textContent !== idleText) {
      $("logBox").textContent = idleText;
    }
    return;
  }
  const lines = [...job.logs];
  if (job.error) {
    lines.push("");
    lines.push(job.error);
  }
  const text = lines.join("\n") || `Job ${job.status}`;
  if ($("logBox").textContent !== text) {
    $("logBox").textContent = text;
    $("logBox").scrollTop = $("logBox").scrollHeight;
  }
}

function renderButtons(project, status) {
  const running = status.job?.status === "running" || status.job?.status === "queued";
  const job = activeJobForProject(status, project);
  const activeJobStatus = job?.status || null;
  const phase = project?.phase;
  const loadedProjectName = project?.name || "";
  const currentProjectName = $("projectName").value.trim();
  const isExistingLoadedProject = Boolean(project?.exists && loadedProjectName === currentProjectName);
  const scenes = project?.state?.scenes || [];
  const hasAllTakes = Boolean(scenes.length && scenes.every((scene) => scene.takes?.length));
  const hasFinal = projectHasFinal(project);
  const complete = isProjectComplete(project);
  const canStartOrContinue = !isExistingLoadedProject
    || (!complete && ["planning", "generating_storyboard", "generating_video", "error"].includes(phase));
  $("startButton").disabled = running || !canStartOrContinue;
  const startLabels = {
    planning: "Start Production",
    generating_storyboard: "Continue Storyboard",
    review_storyboard: "Review Storyboard",
    ready_for_video: "Generate Videos",
    generating_video: "Continue Video Generation",
    review_takes: "Review Takes",
    complete: "Production Complete",
    error: project?.state?.storyboard_approved ? "Retry Video Generation" : "Retry Storyboard",
  };
  const retryLabels = {
    error: scenes.length ? "Retry Production" : "Retry Scene Planning",
    planning: "Retry Planning",
    generating_storyboard: "Retry Storyboard",
    generating_video: "Retry Video Generation",
  };
  if (activeJobStatus === "queued") {
    $("startButton").textContent = "Production Queued";
  } else if (activeJobStatus === "running") {
    $("startButton").textContent = "Production Running";
  } else if (activeJobStatus === "error") {
    $("startButton").textContent = retryLabels[phase] || "Retry Production";
  } else if (activeJobStatus === "finished" && phase !== "complete") {
    $("startButton").textContent = startLabels[phase] || "Continue Production";
  } else {
    $("startButton").textContent = isExistingLoadedProject
      ? (startLabels[phase] || "Start Production")
      : "Start Production";
  }
  $("approveStoryboardButton").disabled = running || phase !== "review_storyboard";
  $("generateVideosButton").disabled = running || phase !== "ready_for_video";
  $("assembleButton").disabled = running || complete || !hasAllTakes;
  $("assembleButton").textContent = complete ? "Final Assembled" : "Assemble Selected Takes";
  $("openFolderButton").disabled = !project?.exists;
  $("openFinalButton").disabled = !hasFinal;
}

function render(status, project) {
  state.status = status;
  state.project = project;
  document.body.classList.toggle("project-complete", isProjectComplete(project));
  if (project?.name) {
    if (project.exists) {
      state.manualNewProject = false;
      rememberSelectedProject(project.name);
    }
    $("projectName").value = project.name;
  }
  if (project?.state?.brief && state.briefProjectName !== project.name) {
    $("briefText").value = project.state.brief;
    state.briefProjectName = project.name;
  }
  renderHealth(status);
  renderLanguage(project, status);
  renderProjectList(status.projects || []);
  const job = activeJobForProject(status, project);
  const activeJobStatus = job?.status || null;
  if (!["queued", "running"].includes(activeJobStatus || "")) {
    state.pendingRegenerationSceneNumber = null;
  }
  const projectViewSignature = JSON.stringify([
    project?.name,
    project?.phase,
    project?.updated_at,
    project?.state?.final_file?.path,
    project?.state?.final_file?.size,
    Object.keys(project?.state?.characters || {}).length,
    project?.state?.style,
    project?.state?.director_notes?.story_grade,
    JSON.stringify(project?.state?.character_references || {}),
    JSON.stringify(project?.state?.project_references || {}),
    JSON.stringify((project?.state?.scenes || []).map((scene) => [
      scene.scene_number,
      scene.keyframe_regeneration_requested_at,
      scene.selected_keyframe,
      scene.keyframe_candidates?.length || 0,
    ])),
    status.job?.project_name,
    status.job?.status,
    JSON.stringify(status.job?.action || {}),
  ]);
  if (projectViewSignature !== state.projectViewSignature) {
    state.projectViewSignature = projectViewSignature;
    renderSummary(project);
    renderProjectReferences(project);
    renderBible(project);
    renderScenes(project, status);
    renderFinal(project);
  }
  renderProgress(project, status);
  renderLogs(status, project);
  renderButtons(project, status);
  const phase = project?.phase || "planning";
  const badgeLabel = activeJobStatus === "error"
    ? "Error"
    : activeJobStatus === "running"
      ? "Running"
      : activeJobStatus === "queued"
        ? "Queued"
        : phaseLabel(phase);
  const badgeTone = activeJobStatus === "error"
    ? "bad"
    : phase === "error"
      ? "bad"
    : phase === "complete"
      ? "ok"
      : "";
  $("phaseBadge").textContent = badgeLabel;
  $("phaseBadge").className = `phase-badge ${badgeTone}`;
  setActiveStep(phase);
}

async function refresh() {
  try {
    const status = await api("/api/status", { method: "GET" });
    const projectName = refreshProjectName(status);
    let project = null;
    if (projectName) {
      project = await api(`/api/projects/${encodeURIComponent(projectName)}`, { method: "GET" });
    }
    render(status, project);
  } catch (error) {
    setMessage(error.message);
  }
}

async function startProduction() {
  try {
    const payload = jobPayload();
    state.manualNewProject = false;
    rememberSelectedProject(payload.project_name);
    setMessage("Starting production...");
    await api("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await refresh();
  } catch (error) {
    setMessage(error.message);
  }
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read the selected reference image."));
    reader.readAsDataURL(file);
  });
}

async function uploadProjectReference() {
  try {
    const projectName = $("projectName").value.trim();
    const brief = $("briefText").value.trim();
    const file = $("referenceFile").files?.[0];
    if (!projectName) {
      throw new Error("Project name is required before adding a reference.");
    }
    if (!file) {
      throw new Error("Choose a reference image first.");
    }
    setMessage("Uploading reference image...");
    const dataUrl = await readFileAsDataUrl(file);
    const payload = {
      project_name: projectName,
      brief,
      is_script: false,
      kind: $("referenceKind").value,
      label: $("referenceLabel").value.trim(),
      filename: file.name,
      data_url: dataUrl,
      output_language: $("outputLanguage").value,
    };
    await api("/api/project-references/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    rememberSelectedProject(projectName);
    state.projectViewSignature = "";
    $("referenceFile").value = "";
    $("referenceLabel").value = "";
    await refresh();
    setMessage("Reference saved. It will guide storyboard keyframe generation.");
  } catch (error) {
    setMessage(error.message);
  }
}

async function deleteProjectReference(referenceId) {
  try {
    const project = $("projectName").value.trim();
    setMessage("Deleting reference...");
    await api(`/api/projects/${encodeURIComponent(project)}/delete-project-reference`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reference_id: referenceId }),
    });
    state.projectViewSignature = "";
    await refresh();
    setMessage("Reference deleted.");
  } catch (error) {
    setMessage(error.message);
  }
}

async function approveStoryboard() {
  try {
    const project = $("projectName").value.trim();
    setMessage("Approving all generated keyframes...");
    await api(`/api/projects/${encodeURIComponent(project)}/approve-storyboard`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    await refresh();
    setMessage("Storyboard approved. Generate videos when ready.");
  } catch (error) {
    setMessage(error.message);
  }
}

async function regenerateKeyframe(sceneNumber) {
  try {
    const project = $("projectName").value.trim();
    state.pendingRegenerationSceneNumber = sceneNumber;
    state.projectViewSignature = "";
    if (state.project && state.status) {
      renderScenes(state.project, state.status);
    }
    const payload = {
      ...jobPayload(),
      scene_number: sceneNumber,
      notes: `Regenerate Scene ${sceneNumber} keyframe from review.`,
    };
    setMessage(`Starting Scene ${sceneNumber} keyframe regeneration...`);
    await api(`/api/projects/${encodeURIComponent(project)}/regenerate-keyframe`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await refresh();
    setMessage(`Scene ${sceneNumber} keyframe regeneration is running.`);
  } catch (error) {
    state.pendingRegenerationSceneNumber = null;
    state.projectViewSignature = "";
    setMessage(error.message);
  }
}

async function generateVideos() {
  try {
    const project = $("projectName").value.trim();
    const payload = jobPayload();
    setMessage("Generating video takes...");
    await api(`/api/projects/${encodeURIComponent(project)}/generate-videos`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await refresh();
  } catch (error) {
    setMessage(error.message);
  }
}

async function selectTake(sceneNumber, takeNumber) {
  try {
    const project = $("projectName").value.trim();
    setMessage(`Selected Scene ${sceneNumber} · Take ${takeNumber}.`);
    await api(`/api/projects/${encodeURIComponent(project)}/select-take`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_number: sceneNumber, take_number: takeNumber }),
    });
    await refresh();
  } catch (error) {
    setMessage(error.message);
  }
}

async function setCharacterReference(characterId, imagePath, sceneNumber, candidateNumber) {
  try {
    const project = $("projectName").value.trim();
    setMessage(`Locking ${characterId} reference...`);
    await api(`/api/projects/${encodeURIComponent(project)}/set-character-reference`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        character_id: characterId,
        path: imagePath,
        source_scene_number: sceneNumber,
        source_candidate: candidateNumber,
      }),
    });
    state.projectViewSignature = "";
    await refresh();
    setMessage(`${characterId} reference locked.`);
  } catch (error) {
    setMessage(error.message);
  }
}

async function clearCharacterReference(characterId) {
  try {
    const project = $("projectName").value.trim();
    setMessage(`Clearing ${characterId} reference...`);
    await api(`/api/projects/${encodeURIComponent(project)}/clear-character-reference`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ character_id: characterId }),
    });
    state.projectViewSignature = "";
    await refresh();
    setMessage(`${characterId} reference cleared.`);
  } catch (error) {
    setMessage(error.message);
  }
}

async function assembleSelectedTakes() {
  try {
    const project = $("projectName").value.trim();
    setMessage("Assembling selected takes...");
    await api(`/api/projects/${encodeURIComponent(project)}/assemble-selected-takes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    await refresh();
    setMessage("Final film is ready.");
  } catch (error) {
    setMessage(error.message);
  }
}

async function openOutputFolder() {
  try {
    const project = $("projectName").value.trim();
    setMessage("Opening output folder...");
    await api(`/api/projects/${encodeURIComponent(project)}/open-output-folder`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    setMessage("Output folder opened.");
  } catch (error) {
    setMessage(error.message);
  }
}

async function openFinalVideo() {
  try {
    const project = $("projectName").value.trim();
    setMessage("Opening final video...");
    await api(`/api/projects/${encodeURIComponent(project)}/open-final-video`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    setMessage("Final video opened.");
  } catch (error) {
    setMessage(error.message);
  }
}

function loadLatest() {
  const latest = state.status?.projects?.[0];
  if (!latest) {
    return;
  }
  state.manualNewProject = false;
  rememberSelectedProject(latest.name);
  $("projectName").value = latest.name;
  refresh();
}

$("startButton").addEventListener("click", startProduction);
$("approveStoryboardButton").addEventListener("click", approveStoryboard);
$("generateVideosButton").addEventListener("click", generateVideos);
$("assembleButton").addEventListener("click", assembleSelectedTakes);
$("openFolderButton").addEventListener("click", openOutputFolder);
$("openFinalButton").addEventListener("click", openFinalVideo);
$("loadLatestButton").addEventListener("click", loadLatest);
$("uploadReferenceButton").addEventListener("click", uploadProjectReference);
$("projectReferenceList").addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) {
    return;
  }
  const target = event.target.closest(".project-reference-delete");
  if (!target) {
    return;
  }
  deleteProjectReference(target.dataset.referenceId);
});
$("sceneList").addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) {
    return;
  }
  const takeTarget = event.target.closest(".take-option");
  if (takeTarget) {
    selectTake(
      Number.parseInt(takeTarget.dataset.sceneNumber, 10),
      Number.parseInt(takeTarget.dataset.takeNumber, 10),
    );
    return;
  }
  const regenerateTarget = event.target.closest(".scene-regenerate");
  if (regenerateTarget) {
    regenerateKeyframe(Number.parseInt(regenerateTarget.dataset.sceneNumber, 10));
    return;
  }
  const referenceTarget = event.target.closest(".reference-option");
  if (referenceTarget) {
    setCharacterReference(
      referenceTarget.dataset.characterId,
      referenceTarget.dataset.imagePath,
      Number.parseInt(referenceTarget.dataset.sceneNumber, 10),
      Number.parseInt(referenceTarget.dataset.candidateNumber, 10),
    );
  }
});
$("bibleGrid").addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) {
    return;
  }
  const target = event.target.closest(".reference-clear");
  if (!target) {
    return;
  }
  clearCharacterReference(target.dataset.characterId);
});
$("newProjectButton").addEventListener("click", () => {
  rememberSelectedProject("");
  state.manualNewProject = true;
  state.languageProjectName = "";
  state.briefProjectName = "";
  $("projectName").value = "";
  $("briefText").value = "";
  $("referenceLabel").value = "";
  $("referenceFile").value = "";
  state.projectViewSignature = "";
  $("briefText").focus();
  refresh();
});

refresh();
state.pollTimer = window.setInterval(refresh, 5000);
