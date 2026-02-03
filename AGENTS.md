# COMMIT DISCIPLINE

- Only commit when:

1. ALL compiler/linter warnings have been resolved

2. The change represents a single logical unit of work

3. Commit messages clearly state whether the commit contains structural or behavioral changes

- Use small, frequent commits rather than large, infrequent ones

# PULL REQUEST Description Guidelines

When creating a pull request, ensure the description adheres to the following:

- Write all PR text related in PT-BR.
- Use github cli (gh) to create/update PRs.
- **Title:** Start the PR title with the relevant description.
- **Summary:** Provide a concise summary of the changes introduced by this PR.
- **Motivation:** Explain the business problem or technical reason behind these changes.
- **Technical Details:** Briefly describe the key technical changes or architectural decisions.
- **Testing Notes:** Include any specific instructions for testing the changes.
- Always return the github PR link in a clickable way.

# CODE QUALITY STANDARDS

- Eliminate duplication ruthlessly

- Express intent clearly through naming and structure

- Make dependencies explicit

- Keep methods small and focused on a single responsibility

- Minimize state and side effects

- Use the simplest solution that could possibly work