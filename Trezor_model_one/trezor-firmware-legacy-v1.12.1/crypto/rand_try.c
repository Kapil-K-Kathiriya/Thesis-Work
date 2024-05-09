#include <stdio.h>
#include "rand.h"

int main() {
    // Reseed the random number generator with a seed value
    // random_reseed(123);

    // Generate and print a random number
    printf("Random number: %u\n", random32());

    // Generate a random buffer and print its contents
    uint8_t buffer[10];
    random_buffer(buffer, sizeof(buffer));
    printf("Random buffer: ");
    for (int i = 0; i < sizeof(buffer); i++) {
        printf("%02x ", buffer[i]);
    }
    printf("\n");

    // // Generate a random number between 0 and 99
    // printf("Random number between 0 and 99: %u\n", random_uniform(100));

    // // Permute a string and print the result
    // char str[] = "hello";
    // random_permute(str, sizeof(str)-1); // sizeof(str) includes null terminator, so subtract 1
    // printf("Permuted string: %s\n", str);

    return 0;
}
