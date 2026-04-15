const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '../../');
const LOGS_DIR = path.join(REPO_ROOT, 'logs');

function walkFiles(dirPath) {
  if (!fs.existsSync(dirPath)) {
    return [];
  }

  return fs.readdirSync(dirPath, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(dirPath, entry.name);

    if (entry.isDirectory()) {
      return walkFiles(entryPath);
    }

    return entry.isFile() ? [entryPath] : [];
  });
}

function getLatestRunFolder() {
  if (!fs.existsSync(LOGS_DIR)) return null;

  const runFolders = fs.readdirSync(LOGS_DIR, { withFileTypes: true })
    .filter(entry => entry.isDirectory() && entry.name.startsWith('run_'))
    .map(entry => entry.name)
    .sort()
    .reverse();

  return runFolders.length > 0 ? runFolders[0] : null;
}

function getAvailablePolicies() {
  const latestRun = getLatestRunFolder();
  if (!latestRun) return [];

  const policiesPath = path.join(LOGS_DIR, latestRun, 'policies');
  if (!fs.existsSync(policiesPath)) return [];

  return fs.readdirSync(policiesPath, { withFileTypes: true })
    .filter(entry => entry.isDirectory())
    .map(entry => entry.name);
}

function getFilesByContext(policyId, contextFolder, fileNamePrefix = '') {
  const latestRun = getLatestRunFolder();
  if (!latestRun) return [];

  const targetDir = path.join(LOGS_DIR, latestRun, 'policies', policyId, contextFolder);
  
  if (!fs.existsSync(targetDir)) {
    return [];
  }

  const allFiles = walkFiles(targetDir).filter((filePath) => filePath.endsWith('.json'));
  
  return allFiles.filter((filePath) => {
    return fileNamePrefix ? path.basename(filePath).startsWith(fileNamePrefix) : true;
  });
}

function getLatestFile(candidates) {
  if (!candidates || candidates.length === 0) {
    return null;
  }
  return candidates.sort().reverse()[0];
}

function getEnvironmentFile() {
  const latestRun = getLatestRunFolder();
  if (!latestRun) return null;

  const environmentPath = path.join(LOGS_DIR, latestRun, 'environment.json');
  if (!fs.existsSync(environmentPath)) {
    return null;
  }

  return environmentPath;
}

module.exports = {
  REPO_ROOT,
  LOGS_DIR,
  walkFiles,
  getLatestRunFolder,
  getAvailablePolicies,
  getFilesByContext,
  getLatestFile,
  getEnvironmentFile
};
