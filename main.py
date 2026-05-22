#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

//2 Вариант()


// структура
typedef struct {
    size_t size;
    int (*comp)(const void*, const void*);       // сравнение
    void (*print)(const void*);                 // вывод
} TypeInf;

// реализация для int
int int_comp(const void* a, const void* b) {     //срав
    return (*(int*)a - *(int*)b);
}
void int_print(const void* a) {                 //вывод
    printf("%d", *(int*)a);
}
TypeInf IntType = { sizeof(int), int_comp, int_print };     //собираем

// реализация для double
int double_comp(const void* a, const void* b) {    //срав
    double diff = (*(double*)a - *(double*)b);
    return (diff > 0) - (diff < 0);              
}
void double_print(const void* a) {                 //вывод
    printf("%.2f", *(double*)a);
}
TypeInf DoubleType = { sizeof(double), double_comp, double_print };   //собираем

//создаем структуру динамич массив
typedef struct {
    void* data;         
    size_t size;        
    size_t capacity;    
    TypeInf* type;     
} DynamicArray;

// Создание массива
DynamicArray* x_mass(TypeInf* type) {
    DynamicArray* x = (DynamicArray*)malloc(sizeof(DynamicArray));
    x->size = 0;           // пустой массив (0 элементов)
    x->capacity = 4;       // вместимость 4 элемента
    x->type = type;        // указатель на тип
    x->data = malloc(x->capacity * type->size);
    return x;
}

// освобождаем память
void xfree(DynamicArray* x) {
    if (x) {
        free(x->data);
        free(x);
    }
}

// добавляем элемент в конец
void xpush(DynamicArray* x, const void* val) {
    if (x->size == x->capacity) {
        x->capacity *= 2;
        x->data = realloc(x->data, x->capacity * x->type->size);
    }
    void* target = (char*)x->data + (x->size * x->type->size);
    memcpy(target, val, x->type->size);
    x->size++;
}

// получаем указатель на элемент по индексу
void* xget(DynamicArray* x, size_t index) {
    if (index >= x->size) return NULL;
    return (char*)x->data + (index * x->type->size);
}

// Выводим массив в консоль
void xprint(DynamicArray* x) {
    printf("[");
    for (size_t i = 0; i < x->size; i++) {
        x->type->print(xget(x, i));
        if (i < x->size - 1) printf(", ");
    }
    printf("]\n");
}

// Сортировка
void xsort(DynamicArray* x) {
    qsort(x->data, x->size, x->type->size, x->type->comp);
}

// Map
DynamicArray* xmap(DynamicArray* x, void (*func)(void*)) {
    DynamicArray* res = x_mass(x->type);
    for (size_t i = 0; i < x->size; i++) {
        void* el = xget(x, i);
        void* temp = malloc(x->type->size);
        memcpy(temp, el, x->type->size); 
        func(temp);                       
        xpush(res, temp);
        free(temp);
    }
    return res;
}

// Where
DynamicArray* xwhere(DynamicArray* x, bool (*predicate)(const void*)) {
    DynamicArray* res = x_mass(x->type);
    for (size_t i = 0; i < x->size; i++) {
        void* el = xget(x, i);
        if (predicate(el)) {
            xpush(res, el);
        }
    }
    return res;
}

// объединяем два массива одинакового типа(контактенация)
DynamicArray* xconcat(DynamicArray* x1, DynamicArray* x2) {
    if (x1->type != x2->type) {
        printf("нельзя склеить массивы разных классов\n");
        return NULL;
    }
    DynamicArray* res = x_mass(x1->type);
    for (size_t i = 0; i < x1->size; i++) xpush(res, xget(x1, i));
    for (size_t i = 0; i < x2->size; i++) xpush(res, xget(x2, i));
    return res;
}


// функции для int
void int_multiply_by_10(void* val) { *(int*)val *= 10; }
bool int_is_even(const void* val)  { return (*(int*)val % 2) == 0; }

// функции для double
void double_add_half(void* val) { *(double*)val += 0.5; }
bool double_is_positive(const void* val) { return *(double*)val > 0.0; }


void test_int_array() {
    printf("\nтест целых чисел\n");
    DynamicArray* arr1 = x_mass(&IntType);
    
    int vals[] = {5, 2, 9, 1, 4, 8};
    for(int i=0; i<6; i++) xpush(arr1, &vals[i]);
    
    printf("ссходный массив: "); xprint(arr1);
    
    xsort(arr1);
    printf("после сортировки: "); xprint(arr1);
    
    DynamicArray* mapped = xmap(arr1, int_multiply_by_10);
    printf("после Map (x * 10):"); xprint(mapped);
    
    DynamicArray* filtered = xwhere(arr1, int_is_even);
    printf("после Where (только четные): "); xprint(filtered);
    
    DynamicArray* arr2 = x_mass(&IntType);
    int extra[] = {100, 200};
    xpush(arr2, &extra[0]); xpush(arr2, &extra[1]);
    
    DynamicArray* concatenated = xconcat(arr1, arr2);
    printf("Конкатенация с [100, 200]: "); xprint(concatenated);
    
    // очистка памяти
    xfree(arr1); xfree(arr2);
    xfree(mapped); xfree(filtered); xfree(concatenated);
}

void test_double_array() {
    printf("\nтестирование вещественных\n");
    DynamicArray* arr1 = x_mass(&DoubleType);
    
    double vals[] = {-1.5, 3.14, 0.0, 2.71, -5.5};
    for(int i=0; i<5; i++) xpush(arr1, &vals[i]);
    
    printf("Исходный массив: "); xprint(arr1);
    
    xsort(arr1);
    printf("После сортировки: "); xprint(arr1);
    
    DynamicArray* mapped = xmap(arr1, double_add_half);
    printf("После Map (x + 0.5): "); xprint(mapped);
    
    DynamicArray* filtered = xwhere(arr1, double_is_positive);
    printf("После Where (x > 0): "); xprint(filtered);
    
    xfree(arr1); xfree(mapped); xfree(filtered);
}

int main() {
    int c = 0;
    while(1) {
        printf("1. запустить тесты для массива целых чисел (int)\n");
        printf("2. Запустить тесты для массива вещественных чисел (double)\n");
        printf("3. Выход\n");
        printf("Ваш выбор: ");
        
        if (scanf("%d", &c) != 1) break;
        
        switch(c) {
            case 1: test_int_array(); break;
            case 2: test_double_array(); break;
            case 3: 
                printf("Выход из программы\n");
                return 0;
            default:
                printf("неправильный выбор, попробуйте еще раз\n");
        }
    }
    return 0;
}
