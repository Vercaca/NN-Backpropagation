gitpush() {
    git add .
    git commit -m "$*"
    git push
}
alias gp=gitpush
