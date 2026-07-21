#define Q36_AGENT_TEST
#define Q36_AGENT_TEST_NO_MAIN
#include "../q36_agent.c"

int main(void) {
    q36_agent_unit_tests_run();
    if (agent_test_failures) {
        fprintf(stderr, "q36-agent tests: %d failure(s)\n",
                agent_test_failures);
        return 1;
    }
    puts("q36-agent tests: ok");
    return 0;
}
