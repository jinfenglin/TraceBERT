import configparser
import logging
import os
import random
import re
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../..")

from git_repo_collector import GitRepoCollector, Commit, Issue
from nltk.tokenize import word_tokenize
import numpy as np


def __read_artifact_dict(file_path, type):
    """
    for issue or commits, 
    return dictionary-like artifacts, such as {commit_id: {commit}}
    """
    
    df = pd.read_csv(file_path)
    df = df.replace(np.nan, regex=True)
    arti = dict()
    
    for index, row in df.iterrows():
        if type == 'commit':
            art = Commit(commit_id=row['commit_id'], summary=row['summary'], \
                diffs=row['diff'], files=row['files'], commit_time=row['commit_time'])
            arti[art.commit_id] = art
        elif type == "issue":
            art = Issue(issue_id=row['issue_id'], desc=row['issue_desc'], \
                comments=row['issue_comments'], create_time=row['created_at'], close_time=row['closed_at'])
            arti[art.issue_id] = art
        else:
            raise Exception("wrong artifact type")
        
    return arti


def __read_artifacts(file_path, type):
    df = pd.read_csv(file_path)
    df = df.replace(np.nan, regex=True)
    arti = []
    for index, row in df.iterrows():
        if type == 'commit':
            art = Commit(commit_id=row['commit_id'], summary=row['summary'], diffs=row['diff'], files=row['files'], commit_time=row['commit_time'])
        elif type == "issue":
            art = Issue(issue_id=row['issue_id'], desc=row['issue_desc'], comments=row['issue_comments'], create_time=row['created_at'], close_time=row['closed_at'])
        elif type == "link":
            iss_id = row["issue_id"]
            cm_id = row["commit_id"]
            art = (iss_id, cm_id)
        else:
            raise Exception("wrong artifact type")
        arti.append(art)
    return arti


def __save_artifacts(art_list, type, output_file):
    df = pd.DataFrame()
    for art in art_list:
        if type == "issue" or type == "commit":
            df = df.append(art.to_dict(), ignore_index=True)
        elif type == "link":
            df = df.append({"issue_id": art[0], "commit_id": art[1]}, ignore_index=True)
        else:
            raise Exception("wrong artifact type")
    df.to_csv(output_file)


def read_artifacts(proj_data_dir):
    commit_file = os.path.join(proj_data_dir, "commit.csv")
    issue_file = os.path.join(proj_data_dir, "issue.csv")
    link_file = os.path.join(proj_data_dir, "link.csv")
    
    issues = __read_artifacts(issue_file, type="issue")
    commits = __read_artifacts(commit_file, type="commit")
    links = __read_artifacts(link_file, type="link")
    return issues, commits, links


def clean_artifacts(proj_dir):
    issue, commit, link = read_artifacts(proj_dir)
    clean_issue_file = os.path.join(proj_dir, "clean_issue.csv")
    clean_commit_file = os.path.join(proj_dir, "clean_commit.csv")
    clean_link_file = os.path.join(proj_dir, "clean_link.csv")

    clean_issues = dict()
    clean_commits = dict()
    clean_links = []

    if not os.path.isfile(clean_issue_file):
        for iss in tqdm(issue):
            if pd.isnull(iss.desc):
                iss.desc = ""
            iss.desc = re.sub("<!-.*->", "", iss.desc)
            iss.desc = re.sub("```.*```", "", iss.desc, flags=re.DOTALL)
            iss.desc = " ".join(word_tokenize(iss.desc))
            iss.comments = " ".join(word_tokenize(iss.comments.split("\n")[0]))  # use only the first comment (title)
            clean_issues[iss.issue_id] = iss
    else:
        tmp_issues = __read_artifacts(clean_issue_file, type="issue")
        for iss in tmp_issues:
            clean_issues[iss.issue_id] = iss

    if not os.path.isfile(clean_commit_file):
        for cm in tqdm(commit):
            diff_sents = eval(cm.diffs)
            if len(diff_sents) < 5:
                continue
            diff_tokens = []
            for sent in diff_sents:
                sent = sent.strip("+- ")
                diff_tokens.extend(word_tokenize(sent))
            cm.diffs = " ".join(diff_tokens)
            cm.summary = " ".join(word_tokenize(cm.summary))
            clean_commits[cm.commit_id] = cm
    else:
        tmp_commit = __read_artifacts(clean_commit_file, type="commit")
        for cm in tmp_commit:
            clean_commits[cm.commit_id] = cm

    for lk in link:
        if lk[0] not in clean_issues or lk[1] not in clean_commits:
            continue
        clean_links.append(lk)

    source_ids = set([x[0] for x in clean_links])
    target_ids = set([x[1] for x in clean_links])

    # remove artifacts do not have associated links
    remove_source = [x for x in clean_issues.keys() if x not in source_ids]
    remove_target = [x for x in clean_commits.keys() if x not in target_ids]
    for rs in remove_source:
        del clean_issues[rs]
    for rt in remove_target:
        del clean_commits[rt]

    # save clean artifacts
    __save_artifacts(clean_issues.values(), type="issue", output_file=clean_issue_file)
    __save_artifacts(clean_commits.values(), type="commit", output_file=clean_commit_file)
    __save_artifacts(clean_links, type="link", output_file=clean_link_file)
    return clean_issues, clean_commits, clean_links


def write_split_chunk(issue, commit, links, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    commit_file = os.path.join(output_dir, "commit_file")
    issue_file = os.path.join(output_dir, "issue_file")
    link_file = os.path.join(output_dir, "link_file")

    sel_issues, sel_commits = [], []
    for iss_id, cm_id in links:
        cm = commit[cm_id]
        iss = issue[iss_id]
        sel_commits.append(cm)
        sel_issues.append(iss)

    __save_artifacts(sel_issues, type="issue", output_file=issue_file)
    __save_artifacts(sel_commits, type="commit", output_file=commit_file)
    __save_artifacts(links, type="link", output_file=link_file)


def split(issue, commit, links, proj_dir):
    train_dir = os.path.join(proj_dir, "train")
    valid_dir = os.path.join(proj_dir, "valid")
    test_dir = os.path.join(proj_dir, "test")

    random.shuffle(links)
    train_pop = int(len(links) * 0.8)
    valid_pop = int(len(links) * 0.1)
    test_pop = int(len(links) * 0.1)

    train_links = links[:train_pop]
    valid_links = links[train_pop: train_pop + valid_pop]
    test_links = links[-test_pop:]

    write_split_chunk(issue, commit, train_links, train_dir)
    write_split_chunk(issue, commit, valid_links, valid_dir)
    write_split_chunk(issue, commit, test_links, test_dir)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel("INFO")
    # projects = ['dbcli/pgcli']
    # projects = ['pallets/flask']
    projects = ['dbcli/pgcli', 'pallets/flask', 'keras-team/keras']

    config = configparser.ConfigParser()
    config.read('credentials.cfg')
    
    output_dir = './data/git_data'

    for repo_path in projects:
        proj_data_dir = os.path.join(output_dir, repo_path)
        
        if not os.path.exists(os.path.join(proj_data_dir, 'issue.csv')):
            # if the issue_csv is not available
            logger.info("Processing repo: {}".format(repo_path))
            git_token = config['GIT']['TOKEN']
            download_dir = 'G:/Document/git_projects'
            rpc = GitRepoCollector(git_token, download_dir, output_dir, repo_path)
            rpc.create_issue_commit_dataset()
        # clean the issue and commits by:
        # remove the stacktrace from issue
        # remove commits with less than 10 lines of changeset
        # remove both issue and commit that have no link associated with we do this for two reasons:
        # 1. link incomplete 2. 10 fold divide
        # split the dataset into 10 fold by links and attach the issue/commit along with those links
        
        clean_issue_file = os.path.join(proj_data_dir, 'clean_issue.csv')
        clean_commits_file = os.path.join(proj_data_dir, 'clean_commit.csv')
        
        if os.path.exists(clean_issue_file) and os.path.exists(clean_commits_file):
            # if the cleaned_issue.csv is available
            clean_issues = __read_artifact_dict(clean_issue_file, 'issue')
            clean_commits = __read_artifact_dict(clean_commits_file, 'commit')
            clean_links_file = os.path.join(proj_data_dir, 'clean_link.csv')
            clean_links = __read_artifacts(clean_links_file, 'link')
        else:
            clean_issue, clean_commits, clean_links = clean_artifacts(proj_data_dir)
        
        split(clean_issues, clean_commits, clean_links, proj_data_dir)