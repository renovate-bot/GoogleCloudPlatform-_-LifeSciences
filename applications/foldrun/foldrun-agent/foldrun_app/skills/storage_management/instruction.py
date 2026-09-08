# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modular instructions for storage management skill."""

STORAGE_MANAGEMENT_INSTRUCTION = """### Storage Management
- **Find orphaned files**: Use find_orphaned_gcs_files to discover ALL GCS files without Agent Platform jobs
  - Scans entire bucket and compares with active Agent Platform jobs
  - Identifies orphaned files from deleted jobs (both pipeline outputs AND FASTA files)
  - Reports total sizes and storage usage
  - Returns separate lists: orphaned_pipeline_dirs and orphaned_fasta_files
  - **IMPORTANT - FASTA File Handling**:
    * NEVER suggest deleting orphaned FASTA files unless user explicitly asks
    * FASTA files are small and may be reused for future submissions
    * Only recommend deleting orphaned_pipeline_dirs by default
    * If user asks to clean up orphans, ONLY include pipeline directories in the cleanup suggestion
    * Example: "Found 34 MB of orphaned pipeline outputs. Would you like to delete them? (Note: 6 orphaned FASTA files will be kept unless you want to remove them too.)"
  - Use this for storage audits and identifying cleanup opportunities
- **Cleanup GCS files**: Use cleanup_gcs_files to delete files in GCS (supports two modes)
  - **Mode 1 - Job-based cleanup**: Provide job_id to find/delete files for a specific job
    - Searches pipeline outputs in timestamped pipeline_runs/ directories
    - By default, does NOT delete FASTA files (include_fasta=false)
    - Only set include_fasta=true if user explicitly asks to delete FASTA files
    - Use this after deleting a job to free up storage space
  - **Mode 2 - Bulk deletion**: Provide gcs_paths list to delete specific directories/files directly
    - Takes the GCS paths from find_orphaned_gcs_files output
    - Supports both directory paths (ending with /) and individual file paths
    - Directory paths are automatically expanded to include all files within
    - **RECOMMENDED WORKFLOW FOR ORPHANED FILES** (ALWAYS USE THIS FOR ORPHANS):
      1. Run find_orphaned_gcs_files to get orphaned_pipeline_dirs and orphaned_fasta_files
      2. Show user the list of orphaned pipeline directories (with sizes)
      3. Ask user if they want to delete the orphaned pipeline directories
      4. Extract the 'path' field from each item in orphaned_pipeline_dirs list
      5. Call cleanup_gcs_files with gcs_paths=[list of directory paths] and search_only=true first
      6. After user confirms, call again with search_only=false and confirm_delete=true
      7. **CRITICAL**: NEVER include orphaned_fasta_files paths unless user EXPLICITLY asks to delete FASTA files
  - Both modes: Two-step workflow (search_only=true to preview, then confirm_delete=true to delete)
  - Returns file paths, sizes in MB/GB, and deletion status

- **CRITICAL: GCS File Cleanup Safety**
  - ALWAYS use two-step workflow: search first, then confirm deletion
  - First call cleanup_gcs_files with search_only=true to show what files exist and their sizes
  - Present the file list and total size to the user
  - Only after user confirms, call again with search_only=false and confirm_delete=true
  - Warn that GCS file deletion is permanent and cannot be undone
  - Suggest keeping FASTA files by default (include_fasta=false) unless user explicitly wants them deleted
"""
