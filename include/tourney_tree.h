#ifndef FAIR_TOPK_TOURNEYTREE_H
#define FAIR_TOPK_TOURNEYTREE_H

#include <algorithm>
#include <vector>
#include <iterator>
#include <limits>
#include <cmath>
#include <concepts>
#include <cassert>

#include "utility.h"
#include "memory.h"

namespace FairTopK {

template <class Compare, class CrossCompute, class Line>
concept LegitTourneyLineTree = requires(Compare compare, CrossCompute crossCompute, Line line, double t) {
    { compare(line, line, t) } -> std::convertible_to<bool>;
    { crossCompute(line, line) } -> std::same_as<double>;
};

template <class EquivCompare, class Func, class Line>
concept LegitApplyToEquivs = requires(EquivCompare equivCompare, Func func, Line line) {
    { equivCompare(line, line) } -> std::convertible_to<bool>;
    { func(line) } -> std::same_as<void>;
};

template <class Line, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
class KineticTourneyLineTree {
public:
    KineticTourneyLineTree(std::vector<Line>::const_iterator begin, std::vector<Line>::const_iterator end, 
        double beginTime, double endTime, Compare compare, CrossCompute crossTimeCompute);
    bool replaceTop(const Line& line);
    const Line& Top() const { return lines[nodes[0].topLineIdx]; }
    double getNextEventTime() const { return nodes[0].nextEventTime; }
    bool Advance();
    template<class EquivCompare, class Func> requires LegitApplyToEquivs<EquivCompare, Func, Line>
    void applyToTopEquivs(EquivCompare equivCompare, Func func) const;

    ~KineticTourneyLineTree() { FairTopK::freeAligned(nodes); }
    KineticTourneyLineTree(const KineticTourneyLineTree&) = delete;
    KineticTourneyLineTree(KineticTourneyLineTree&&) = delete;
    KineticTourneyLineTree& operator=(const KineticTourneyLineTree&) = delete;
    KineticTourneyLineTree& operator=(KineticTourneyLineTree&&) = delete;
private:
    enum class EventType : unsigned int { Top = 1, Left = 2, Right = 3 };
    enum class TopType : unsigned int { Left = 4, Right = 8 };
    struct Node {
        Node *getLeft() const noexcept { return (Node *)this + 1; /* Pre-order traversal layout */ }
        Node *getRight() const noexcept { return rightChild; }
        void setSecondChild(Node *right) noexcept {
            rightChild = right;
        }
        EventType getEventType() const noexcept {
            return EventType(flags & eventMask);
        }
        void setEventType(EventType type) noexcept {
            flags &= (~eventMask); 
            flags |= (unsigned int)type;
        }
        TopType getTopType() const noexcept {
            return TopType(flags & topMask);
        }
        void setTopType(TopType type) noexcept {
            flags &= (~topMask);
            flags |= (unsigned int)type;
        }
        void updateNextEvent() noexcept {
            assert(!isLeaf());

            nextEventTime = topChangeTime;
            EventType type = EventType::Top;
            Node *left = getLeft();
            Node *right = getRight();

            if (left->nextEventTime < nextEventTime) {
                nextEventTime = left->nextEventTime;
                type = EventType::Left;
            }
            if (right->nextEventTime < nextEventTime) {
                nextEventTime = right->nextEventTime;
                type = EventType::Right;
            }
            setEventType(type);
        }
        bool isLeaf() const noexcept { return rightChild == nullptr; /* Full tree */ }

        int topLineIdx = -1;
    private:
        unsigned int flags = 0;
        Node *rightChild = nullptr;
    public:
        double topChangeTime;
        double nextEventTime;
    private:
        static constexpr unsigned int eventMask = 0x3;
        static constexpr unsigned int topMask = 0xC;
    };
    
    Node *recursiveBuild(std::vector<Line>::const_iterator begin, std::vector<Line>::const_iterator end, int& idx);
    void recursiveReplaceTop(const Line& line, Node *node);
    bool recursiveAdvance(Node *node);

    void updateNodeTop(int leftTopLineIdx, int rightTopLineIdx, Node *node);
    template<class EquivCompare, class Func> requires LegitApplyToEquivs<EquivCompare, Func, Line>
    void applyToTopEquivs(const Line& topLine, const Node *node, EquivCompare&& equivCompare, Func&& func) const;

    Compare compare;
    CrossCompute crossTimeCompute;
    double curTime = 0.0;
    double endTime = 0.0;

    std::vector<Line> lines;
    Node *nodes = nullptr;
};

template <class Line, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
inline void KineticTourneyLineTree<Line, Compare, CrossCompute>::updateNodeTop(int leftTopLineIdx, int rightTopLineIdx, 
    Node *node) {
    const Line &topLeft = lines[leftTopLineIdx];
    const Line &topRight = lines[rightTopLineIdx];

    bool futureExchange = false;
    if (compare(topLeft, topRight, curTime)) {
        node->topLineIdx = leftTopLineIdx;
        node->setTopType(TopType::Left);
        futureExchange = compare(topRight, topLeft, endTime);
    }
    else {
        node->topLineIdx = rightTopLineIdx;
        node->setTopType(TopType::Right);
        futureExchange = compare(topLeft, topRight, endTime);
    }

    if (futureExchange) {
        node->topChangeTime = std::clamp(crossTimeCompute(topLeft, topRight), curTime, endTime);
    }
    else {
        node->topChangeTime = std::numeric_limits<double>::infinity();
    }
}

template <class Line, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
KineticTourneyLineTree<Line, Compare, CrossCompute>::KineticTourneyLineTree(std::vector<Line>::const_iterator begin, std::vector<Line>::const_iterator end, 
    double beginTime, double endTime, Compare compare, CrossCompute crossTimeCompute) : 
    curTime(beginTime), endTime(endTime), compare(std::move(compare)), crossTimeCompute(std::move(crossTimeCompute)) {
    auto lineCount = std::distance(begin, end);
    nodes = FairTopK::allocAligned<Node>(2 * lineCount - 1);

    lines.reserve(lineCount);
    std::copy(begin, end, std::back_inserter(lines));

    int idx = 0;
    recursiveBuild(lines.cbegin(), lines.cend(), idx);
}

template <class Line, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
KineticTourneyLineTree<Line, Compare, CrossCompute>::Node *KineticTourneyLineTree<Line, Compare, CrossCompute>::recursiveBuild(
    std::vector<Line>::const_iterator begin, std::vector<Line>::const_iterator end, int& idx) {
    if (begin == end) return nullptr;

    Node *node = new (&nodes[idx++]) Node();

    auto dis = std::distance(begin, end);
    if (dis <= 1) {
        node->topLineIdx = std::distance(lines.cbegin(), begin);
        node->topChangeTime = std::numeric_limits<double>::infinity();
        node->nextEventTime = std::numeric_limits<double>::infinity();
        return node;
    }

    auto mid = begin + dis / 2;
    Node *left = recursiveBuild(begin, mid, idx);
    Node *right = recursiveBuild(mid, end, idx);

    node->setSecondChild(right);
    
    updateNodeTop(left->topLineIdx, right->topLineIdx, node);

    node->updateNextEvent();

    return node;
}

template <class Line, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
bool KineticTourneyLineTree<Line, Compare, CrossCompute>::replaceTop(const Line& line) {
    Node *root = &nodes[0];

    int curTopLineIdx = root->topLineIdx;
    lines[curTopLineIdx] = line;
    if (!root->isLeaf())  [[likely]]
        recursiveReplaceTop(line, root);

    return root->topLineIdx != curTopLineIdx;
}

template <class Line, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
void KineticTourneyLineTree<Line, Compare, CrossCompute>::recursiveReplaceTop(const Line& line, Node *node) {
    assert(!node->isLeaf());

    Node *left = node->getLeft();
    Node *right = node->getRight();

    TopType topType = node->getTopType();

    if (topType == TopType::Left) {
        if (!left->isLeaf()) recursiveReplaceTop(line, left);
    }
    else {
        if (!right->isLeaf()) recursiveReplaceTop(line, right);
    }

    updateNodeTop(left->topLineIdx, right->topLineIdx, node);

    node->updateNextEvent();
}

template <class Line, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
bool KineticTourneyLineTree<Line, Compare, CrossCompute>::Advance() {
    Node *root = &nodes[0];
    curTime = root->nextEventTime;

    if (root->isLeaf()) [[unlikely]]
        return false;

    return recursiveAdvance(root);
}

template <class Line, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
bool KineticTourneyLineTree<Line, Compare, CrossCompute>::recursiveAdvance(Node *node) {
    assert(!node->isLeaf());

    bool topUpdated = false;
    Node *left = node->getLeft();
    Node *right = node->getRight();
    EventType eventType = node->getEventType();
    if (eventType == EventType::Top) {
        bool isLeft = (node->getTopType() == TopType::Left);

        TopType newTopType = isLeft ? TopType::Right : TopType::Left;
        node->topLineIdx = isLeft ?  right->topLineIdx : left->topLineIdx;
        node->topChangeTime = std::numeric_limits<double>::infinity();
        node->setTopType(newTopType);

        node->updateNextEvent();
        topUpdated = true;
    }
    else {
        bool childTopUpdated = false;
        if (eventType == EventType::Left) {
            if (!left->isLeaf()) childTopUpdated = recursiveAdvance(left);
        }
        else {
            if (!right->isLeaf()) childTopUpdated = recursiveAdvance(right);
        }

        if (childTopUpdated) {
            int preTopLineIdx = node->topLineIdx;

            updateNodeTop(left->topLineIdx, right->topLineIdx, node);

            if (node->topLineIdx != preTopLineIdx) topUpdated = true;
        }

        node->updateNextEvent();
    }

    return topUpdated;
}

template <class Line, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
template <class EquivCompare, class Func> requires LegitApplyToEquivs<EquivCompare, Func, Line>
void KineticTourneyLineTree<Line, Compare, CrossCompute>::applyToTopEquivs(EquivCompare equivCompare, Func func) const {
    const Node *root = &nodes[0];
    const Line &topLine = lines[root->topLineIdx];
    func(topLine);

    if (!root->isLeaf()) [[likely]]
        applyToTopEquivs(topLine, root, std::move(equivCompare), std::move(func));
}

template <class Line, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
template <class EquivCompare, class Func> requires LegitApplyToEquivs<EquivCompare, Func, Line>
void KineticTourneyLineTree<Line, Compare, CrossCompute>::applyToTopEquivs(const Line& topLine, const Node *node, 
    EquivCompare&& equivCompare, Func&& func) const {
    assert(!node->isLeaf());

    const Node *curNode = node;
    do {
        const Node *left = curNode->getLeft();
        const Node *right = curNode->getRight();

        if (curNode->getTopType() == TopType::Left) {
            const Line& rightLine = lines[right->topLineIdx];
            if (std::forward<EquivCompare>(equivCompare)(topLine, rightLine)) {
                std::forward<Func>(func)(rightLine);
                if (!right->isLeaf())
                    applyToTopEquivs(topLine, right, std::forward<EquivCompare>(equivCompare), std::forward<Func>(func));
            }
            curNode = left;
        }
        else {
            const Line& leftLine = lines[left->topLineIdx];
            if (std::forward<EquivCompare>(equivCompare)(topLine, leftLine)) {
                std::forward<Func>(func)(leftLine);
                if (!left->isLeaf()) 
                    applyToTopEquivs(topLine, left, std::forward<EquivCompare>(equivCompare), std::forward<Func>(func));
            }
            curNode = right;
        }
    } while (!curNode->isLeaf());
}

}

#endif
